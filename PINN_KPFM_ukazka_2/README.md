# PINN pro časové profily KPFM

`kpfm_pinn.py` načítá `data/RH*.dat` a trénuje samostatný 2D PINN pro každou relativní vlhkost. Síť aproximuje potenciál φ(x, y, t) a kalibraci KPFM `V_KPFM = a·φ + b`. Koeficienty a, b jsou společné pro celou časovou řadu dané RH. Nezávislá měření na obou elektrodách je ukotvují k napájecím napětím.

**Pro odhad D doporučuji režim `--reference-d`.** Nezávislý numerický inverzní řešič nejdřív odhadne D přímo z časových řezů při stejných fyzikálních předpokladech; PINN pak s tímto D rekonstruuje potenciál a plní PDE i okrajové loss. Bez tohoto přepínače se D učí společně s vahami sítě. Tento čistý inverzní PINN je experimentální: při ověření se podařilo fitovat potenciál, ale nepodařilo se spolehlivě obnovit známé D. Jeho číselný odhad zatím nepovažujte za ověřený výsledek. Oba režimy ukládají metodu odhadu do `summary.json`.

## Spuštění v PowerShellu

Místní `.venv` obsahuje CPU PyTorch a používá již dostupné NumPy, SciPy a Matplotlib:

```powershell
# Kontrola všech dat bez trénování, funguje i bez PyTorch:
python kpfm_pinn.py inspect

# Samotný numerický odhad D pro všechny RH, také bez PyTorch:
python kpfm_pinn.py reference

# Jedna vlhkost s nezávislým odhadem D a následnou rekonstrukcí PINN:
.\.venv\Scripts\python.exe kpfm_pinn.py train --rh 70 --reference-d

# Všechny vlhkosti; nezávislé inicializace pro kontrolu stability D:
.\.venv\Scripts\python.exe kpfm_pinn.py train --reference-d --seeds 7 19 31 --lbfgs-steps 300

# Vynechat celou časovou křivku z trénování a změřit predikční chybu:
.\.venv\Scripts\python.exe kpfm_pinn.py train --rh 80 --reference-d --holdout-time-index 3

# Syntetická data z nezávislého řešení konečnými diferencemi:
.\.venv\Scripts\python.exe kpfm_pinn.py demo --reference-d --steps 6000 --output outputs/demo

# Testy načítání, geometrie a fyzikálního rezidua/inverze D:
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Pro nové prostředí:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Výchozí trénink má 6 000 kroků Adam, tři skryté vrstvy po 48 neuronech s tanh. Prvních 2 000 kroků drží D pevné, aby nesprávná počáteční křivost sítě nestlačila D předčasně k nule. Fyzikální loss se během druhé poloviny této fáze postupně zapíná; první fáze fituje především pozorovaný časový průběh. `--parameter-warmup` nastavuje tuto fázi (nejvýše polovinu celkových kroků). Polovina kolokačních bodů cílí do mezery a zbytek rovnoměrně pokrývá dielektrikum; poměr mění `--gap-fraction`. Časové body kombinují rovnoměrné rozložení a zahuštění u t=0, aby síť neobcházela rovnici rychlou změnou mezi prvními snímky. Volitelný L-BFGS používá pevnou sadu kolokačních bodů.

Tři fyzikální vstupy x, y, t se uvnitř sítě rozšiřují o lokálně škálované x, y a logaritmickou souřadnici času. Tyto pomocné souřadnice umožňují lépe zachytit úzkou mezeru a rychlý počáteční přechod. Automatická diferenciace zahrnuje i derivace tohoto převodu; PDE se stále derivuje podle původních bezrozměrných x, y, t.

Počáteční D je podle očekávání pomalého transportu nastavené na 10⁻¹⁶ m²/s. Je to inicializace, nikoli omezení nebo výsledek. Pro kontrolu závislosti na inicializaci změňte `D_initial_m2_s` v kopii konfigurace a předejte ji přes `--config`. Syntetické demo používá vlastní počáteční D=10⁻¹⁴ m²/s a známou pravdu 2×10⁻¹⁴ m²/s.

Referenční řešič v `kpfm_reference.py` sestaví diskrétní Laplaceův operátor na celé 2D geometrii, zahrne pevná napětí elektrod a reflektující vnější okraje a řeší časový vývoj pomocí vlastní báze operátoru. Fit D, a, b používá všechny nevynechané časové řezy, více počátečních D a rozsah D=10⁻²² až 10⁻¹⁰ m²/s. Pokud optimum skončí na konci rozsahu, výsledek je pouze mez a skript přidá varování. Předpoklad malého D ho nijak nefixuje. `--reference-grid 31` je výchozí prostorová mřížka; pro kontrolu diskretizace porovnejte s `--reference-grid 41` či 51. Rozměry elektrod jsou na těchto mřížkách zarovnané. Na velmi hrubé mřížce je náboj/potenciál u hran jen aproximace.

V režimu `--reference-d` není D výstupem optimalizace sítě a žádný rozdíl mezi referenčním a neuronovým fitem se neskrývá: `summary.json` ukládá úplný numerický výsledek, metodu odhadu i nezávislé reziduum PINN. Referenční inverze má stejná omezení fyzikálního modelu a neznámé počáteční mapy jako PINN. Samotná numerická přesnost je neodstraňuje.

Přepínače `--collocation`, `--initial-points`, `--boundary-points`, `--steps`, `--width`, `--depth`, `--seed`, `--lr` a `--parameter-lr` umožňují měnit rozlišení a optimalizaci. `--device cuda` vyžaduje vlastní instalaci PyTorch s CUDA. Krátké spuštění je pouze kontrola funkčnosti; počet kroků sám o sobě nezaručuje konvergenci.

## Data a geometrie

Konfigurace je v `pinn_config.json`. Hodnoty vycházejí z upřesnění experimentu:

| Veličina | Nastavení |
|---|---|
| Povrch | čtverec [0,10] × [0,10] µm |
| Levá elektroda | [0,4] × [4,6] µm, 2 V |
| Pravá elektroda | [6,10] × [4,6] µm, 0 V |
| Měřicí řez | y = 5 µm, x_soubor + 2 µm |
| Backgate | Si, 0 V |
| Tloušťka SiO₂ | pracovní hodnota 300 nm; experiment uvádí nejméně 300 nm |
| Relativní permitivita | pracovní hodnota 3,9 |
| Počáteční stav při zapnutí | pouze levá elektroda na 2 V; dielektrikum a pravá elektroda na 0 V v transportním modelu |
| Prodleva prvního měření | fitovaná v intervalu 0–3 s podle upřesnění experimentu |

Elektrody jsou konečné obdélníky, nikoli celé levé/pravé hrany čtverce. PDE platí pouze na dielektriku mimo ně. Okrajové body elektrod pokrývají čela i horní a dolní strany. Síť uvnitř kovu nemá fyzikální význam; export tam dosazuje předepsané napětí a neuvádí povrchový náboj dielektrika.

Všech sedm souborů obsahuje 256 poloh a sedm časů. RH80 má časy 0–30 s po 5 s. Ostatní RH mají 0; 0,5; 1; 2,5; 5; 7,5; 10 min. RH50 a RH70 mají desetinnou čárku a chybějící hlavičku; jejich časy jsou výslovně uložené v konfiguraci podle potvrzení uživatele. První číselný řádek se zachovává. Skript hlídá počet sloupců, jednotky času, konečné hodnoty a rostoucí x i t.

## Fyzikální model a loss

První implementace používá **efektivní model s lokální kapacitou vůči backgate**:

\[
C_A=\varepsilon_0\varepsilon_r/d,\qquad
\sigma_{\rm eff}=C_A(\phi-V_{\rm gate}),\qquad
\partial_t\sigma_{\rm eff}=D_{\rm eff}\nabla^2_{xy}\sigma_{\rm eff}.
\]

Pro konstantní kapacitu a konstantní V_gate tedy řešíme

\[
\partial_t\phi=D_{\rm eff}(\phi_{xx}+\phi_{yy}).
\]

Jde o pracovní aproximaci difuze/rozlévání potenciálu na kapacitně vázaném povrchu. V modelu odporové vrstvy by `D_eff = vodivost_na_čtverec / C_A`. Tento koeficient nelze automaticky ztotožnit s mikroskopickým difuzním koeficientem iontů. Drift jednotlivých nosičů v elektrickém poli, zachytávání, únik do objemu a nelineární elektrochemie zde nejsou samostatně modelovány. Pokud jsou podstatné, bude potřeba rozšířit transportní i pozorovací model.

Normalizace: ξ=x/L, η=y/L, τ=t/T a u=(φ−V_R)/(V_L−V_R). Síť se učí `alpha = D_eff·T/L² = exp(log_alpha)` a fyzikální reziduum je `r = u_τ − alpha·(u_ξξ + u_ηη)`. L je vždy **10 µm**, nikoli délka měřeného řezu; T je délka konkrétní časové řady. Druhé derivace počítá automatická diferenciace. Výstup D je v m²/s i µm²/s.

Celková loss je vážený součet:

| Člen | Co penalizuje |
|---|---|
| `data` | odchylku `a·φ+b` od KPFM na dielektriku |
| `calibration` | odchylku `a·V_elektrody+b` od měření na kovu |
| `pde` | reziduum difuzní rovnice ve 2D dielektriku |
| `electrode` | u=1 vlevo a u=0 vpravo na všech exponovaných stranách kovu |
| `outer_noflux` | ∂u/∂n=0 na vnějším obvodu mimo kov |
| `initial` | nulový počáteční potenciál a efektivní náboj na dielektriku; elektrody mají předepsaná napětí |

**Okrajové podmínky vstupují přes loss**, síť je nemá vestavěné do své architektury. Nulový tok přes vnější čtverec je předpoklad uzavřené oblasti, nikoli experimentálně potvrzený okraj vzorku. Jestli jde pouze o výřez většího povrchu, je nutné otestovat jeho vliv nebo rozšířit oblast.

Výchozí `initial_condition: "left_electrode_only"` předepisuje v okamžiku zapnutí na celém dielektriku φ=V_gate=0, tedy σ_eff=0 v rámci lokální kapacitní aproximace. Pouze levý kovový obdélník je na 2 V; pravý je na 0 V. První měřený řez se již **nerozšiřuje do plochy** a není počáteční podmínkou. Je to pozorování, vůči kterému se kontroluje model. Numerický řešič i PINN používají společnou funkci `initial_envelope`, která pro tento režim vrací nulu. Obrázek `outputs/figures/pocatecni_stav.png` ukazuje přesný počátek, shodný pro všechny RH.

Čas souboru je čas relativně k prvnímu snímku. Referenční inverze fituje navíc prodlevu τ₀ v povoleném intervalu 0–3 s: `t_sim = t_soubor + τ₀`. Počáteční podmínka platí v t_sim=0, a nikoli ve fitovaném okamžiku prvního snímku. Výstupy obsahují oba časy. PINN v režimu `--reference-d` převezme D i τ₀ z numerického fitu; samotný experimentální PINN používá pevné `measurement_time_offset_s` z konfigurace. `max_measurement_time_offset_s: 0` vypne fit prodlevy.

Nulový náboj na dielektriku obecně neznamená nulový elektrostatický potenciál od nabité elektrody. Dosavadní transportní model toto okamžité nelokální pole samostatně neřeší. Pokud proto nedokáže popsat první nenulový KPFM řez ani při maximální prodlevě 3 s, nesmí se počáteční mapa uměle dobít na celou levou stranu. Nesoulad zůstává ve fitu a je důvodem doplnit elektrostatický pozorovací model. `first_profile_RMSE_V`, `time_offset_at_upper_bound` a `fit_warning` jej pomáhají rozpoznat.

Původní režimy `extrude_first_profile` a `electrode_envelope_first_profile` zůstávají dostupné pro reprodukci starých pokusů a syntetického dema. Pro opravené experimentální zadání se nepoužívají. Orientační obrázek RH60 není použitý pro trénink, kalibraci ani odhad D.

## Jak interpretovat náboj a D

`sigma_eff_C_m2` je **efektivní plošný náboj lokálního kondenzátoru**, nikoli jednoznačná rekonstrukce zachyceného či mobilního povrchového náboje. Export obsahuje také změnu vůči prvnímu profilu `delta_sigma_eff_C_m2`. Pro d=300 nm je C_A přibližně 1,15×10⁻⁴ F/m². Protože skutečná tloušťka je jen zdola omezená, náboj škáluje jako 300 nm / d. Platí to při zachování ostatních předpokladů modelu.

Samotná prostorová křivost KPFM potenciálu není plošný náboj: použití `−ε·∇²_xy φ` by bez správného 3D elektrostatického modelu zaměnilo povrchový a objemový náboj. Přesná rekonstrukce náboje vyžaduje zahrnout elektrostatickou geometrii, okrajová pole a přenosovou funkci sondy, případně referenční stav bez zachyceného náboje. Zde tato fyzika vstupuje pouze aproximací C_A a empirickým převodem a, b.

Z jediné přímky není obecná 2D počáteční distribuce ani její vývoj jednoznačně určitelný. Výsledné D je podmíněné zvoleným počátečním rozšířením, uzavřeným vnějším okrajem a transportním modelem. Nízká loss sama o sobě nedokazuje identifikovatelnost D. Porovnejte několik seedů, časové holdouty, jiná počáteční D a hustší kolokace; u hranic a počáteční mapy je vhodná další fyzikální studie citlivosti. Rozptyl seedů není statistický interval spolehlivosti. Hodnoty D se mezi RH nesdílejí a skript nevnucuje závislost na vlhkosti.

## Výstupy

Obrázky všech RH lze znovu vytvořit bez trénování sítě:

```powershell
.\.venv\Scripts\python.exe plot_kpfm.py
```

Skript použije uložené výsledky `outputs/reference_all/reference_results.json`, včetně původní fyzikální konfigurace a rozlišení numerické mřížky. V `outputs/figures/` vytvoří společné přehledy `prehled_mapy.png` a `prehled_rezy.png` a pro každou RH složku se soubory `mapy_kpfm.png`, `mapy_potencial.png`, `mapy_naboj.png` a `rezy.png`. Mapy obsahují všechny časy měření. Řezy porovnávají měření a model v y=5 µm, ukazují příčný modelový řez v x=5 µm a odchylky od měření. Pro každou veličinu je barevná stupnice společná všem vybraným RH a časům. Zdrojová čísla jsou v NPZ a CSV; vše je také zabaleno v `outputs/figures_obrazky.zip`.

Tyto obrázky pocházejí z numerického referenčního řešiče, nikoli z neuronové sítě nebo měřených 2D map. Používají opravený počáteční stav pouze s levou elektrodou a prodlevu nejvýše 3 s. V titulcích jednotlivých časů je uveden čas měření i čas od zapnutí. Předchozí chybné mapy a výsledky s plošně rozšířeným profilem jsou zachované v `outputs/history_before_electrode_initial/` a nejsou aktuálními výstupy.

Každý běh je v `outputs/pinn/RH…/seed…/` (opakované spuštění stejného cíle přepíše jeho výsledky):

- `summary.json`: D, kalibrace, RMSE dat a elektrod, nezávislé kontroly PDE a okrajů, použité předpoklady.
- `profiles.csv`: naměřený a fitovaný KPFM signál, modelový potenciál, efektivní náboj a jeho změna v měřeném řezu; příznaky holdoutu a dielektrika.
- `fields.npz`: 2D potenciály a efektivní náboj v časech měření, souřadnice a maska dielektrika. Mimo měřenou přímku jde o podmíněnou rekonstrukci.
- `loss.csv`, `fit_and_loss.png`, `potential_maps.png`: jednotlivé nevážené členy loss, shoda s profily a mapy potenciálu.
- `model.pt`, `run_config.json`: váhy sítě a úplné nastavení pro reprodukci.

`outputs/pinn/results.json` shromažďuje výsledky vybraných RH a seedů. `inspect` ukládá `data_summary.json`. Diagnostika vyhodnocuje PDE na nezávislé husté prostorově časové mřížce a okraje na nové sadě bodů. Hodnoty rezidua jsou bezrozměrné, chyba potenciálu elektrod ve V. Vynechaný čas se nepoužívá ani ke kalibraci. Rozhraní rezervuje první čas pro kontrolu počátku a umožňuje vynechat až následující časové profily.

`summary.json` obsahuje `warnings`, pokud D klesne pod 1 % počátečního odhadu nebo vzroste nad jeho stonásobek. Jsou to upozornění na extrémní pohyb parametru, nikoli fyzikální meze ani test identifikovatelnosti. Při malých časových změnách nebo nesouladu transportního modelu s daty může D kolabovat k nule a přesto vytvořit dobrý fit potenciálu. Tento stav se nesmí prezentovat jako změřený difuzní koeficient.

Demo uvádí známé D i relativní chybu odhadu a při chybě nad 20 % přidá výslovné varování. Dva kroky demo tréninku stačí ke kontrole numerické inverze, nikoli ke kontrole neuronové rekonstrukce. Testy ověřují také obnovení známého D nezávislou 2D numerickou inverzí s celým časem vynechaným z fitu.

## Dosavadní ověření

Devět automatických testů kontroluje načítání, geometrii, derivace, inverzi D, přesně nulový počáteční stav dielektrika a obnovu známé prodlevy 1,8 s. Syntetické demo s původním měřeným počátečním profilem je oddělený test; dříve obnovilo známé D s chybou přibližně 1 % na odlišných mřížkách. To neověřuje fyzikální správnost počátečního modelu pro experimentální KPFM.

Aktuální výsledky všech RH jsou v `outputs/reference_all/reference_results.csv` a `.json`. Po opravě počátku dosahuje fit prodlevy u všech RH horní povolené meze 3 s a chyba prvního profilu je přibližně 0,07–0,12 V. D proto zůstává pracovním parametrem čistě difuzního modelu; nevyřešený počáteční elektrostatický signál se nesmí vydávat za transportní náboj. Staré hodnoty D ani stará kontrola mřížky se na tuto opravenou konfiguraci nevztahují.

Starý kontrolní PINN běh `outputs/reference_example` používal původní chybný počátek a není aktuálním výsledkem. Krátký integrační běh s opraveným počátkem a časovým posunem je v `outputs/electrode_initial_smoke`; ověřuje funkční průchod kódem, nikoli konvergenci neuronové rekonstrukce.

## Podklady k předpokladům

- [PyTorch: automatické derivace a create_graph](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html).
- [Quantifying nanoscale charge density features of contact-charged surfaces with an FEM/KPFM-hybrid approach](https://arxiv.org/abs/2209.01889): převod KPFM na náboj závisí na elektrostatické geometrii a odezvě sondy.
- [The weight function for charges — A rigorous theoretical concept for Kelvin probe force microscopy](https://users.aalto.fi/~asf/publications/weight_kpfm.pdf): závislost signálu na náboji, kapacitách a geometrii měření.

Geometrie implementace vychází z upřesnění v rozhovoru. Poslední slide místního souboru `PINN pro fyzikální inženýry.pptx` uvádí téma rekonstrukce potenciálu a odkaz na praktikum; rozměrové schéma v něm není.
