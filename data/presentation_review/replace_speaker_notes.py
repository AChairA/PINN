from pathlib import Path
import copy
import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


notes = {
    1: [
        'Dnes si ukážeme, jak neuronové síti zadat fyzikální rovnici.',
        'Nejdřív projdeme princip na jednoduchém potenciálu. Potom vyřešíme chladnutí tyče a odhadneme neznámý parametr.',
        'Kód budeme sledovat v notebooku PINN_pruvodce.ipynb.',
    ],
    2: [
        'Představme si, že máme jen několik měření a chceme znát průběh mezi nimi.',
        'Samotná data často nestačí. Známe ale rovnici, kterou musí řešení splňovat.',
        'Síti proto zadáme data, rovnici a podmínky na okrajích.',
        'Můžeme hledat samotný průběh, nebo neznámý fyzikální parametr.',
        'Tady budeme používat známý tvar rovnice. Její tvar se síť sama učit nebude.',
    ],
    3: [
        'Neuronová síť je funkce. Dostane vstup a vrátí výstup.',
        'Vstupem může být poloha a čas. Výstupem může být teplota nebo potenciál.',
        'Váhy a posuny určují tvar této funkce. Společně je označujeme θ.',
        'Ztrátová funkce říká, jak moc je aktuální výsledek špatně.',
        'Při trénování měníme parametry tak, aby tato chyba klesala.',
    ],
    4: [
        'U běžného učení porovnáváme výstup s naměřenými hodnotami.',
        'U PINN navíc kontrolujeme, jestli výstup splňuje fyzikální rovnici.',
        'Všechny členy rovnice převedeme na jednu stranu. Pro přesné řešení musí vyjít nula.',
        'To, co po dosazení našeho odhadu vyjde místo nuly, je reziduum. Je to chyba rovnice v daném bodě.',
        'Tady je reziduum druhá derivace potenciálu plus ρ/ε₀.',
        'Chyby umocníme na druhou a sečteme. Tím dostaneme fyzikální ztrátu.',
    ],
    5: [
        'Pro Poissonovu rovnici potřebujeme druhou derivaci podle polohy.',
        'PyTorch ji spočítá automaticky ze zadané funkce. Neděláme rozdíly mezi sousedními body.',
        'V kódu ukaž dvě volání autograd.grad: první počítá první derivaci, druhé druhou derivaci.',
        'create_graph=True umožní přes tyto derivace dál počítat gradient ztráty.',
        'backward pak řekne, jak změna váhy ovlivní ztrátu ze všech pěti bodů.',
        'Tady váhu ještě neměníme. Jen počítáme chybu a její gradient.',
    ],
    6: [
        'Samotná rovnice nestačí. Musíme také říct, jaké hodnoty chceme na koncích.',
        'Tady chceme vlevo potenciál 1 a vpravo 2.',
        'První možnost: přidáme chybu na okrajích do celkové ztráty.',
        'Pozor: samotný tanh(wx) tyto hodnoty nikdy nesplní. V nule dává nulu a hodnoty 2 nedosáhne. Horní kód jen ukazuje sestavení ztráty.',
        'Druhá možnost: správné hodnoty vložíme přímo do vzorce. To dělá spodní kód.',
        'Člen 1+x nastaví okraje. Korekce x(1−x) je na obou koncích nulová, takže je nezkazí.',
        'Tento trik zaručuje hodnoty na okrajích. Pro předepsané derivace potřebujeme jiný tvar.',
        'Pokud padne dotaz na aktivaci: se čtvercovou aktivací lze přesné řešení vyjádřit. Musíme ale vstup i výstup také násobit a posouvat. Pouhá výměna tanh za druhou mocninu nestačí.',
    ],
    7: [
        'Máme tyč dlouhou dva metry. Oba konce držíme na 20 °C, střed nejdřív zahřejeme na 450 °C.',
        'Pak ohřev vypneme a sledujeme dvě hodiny chladnutí.',
        'Síť dostane polohu a čas a vrátí teplotu. Tepelnou difuzivitu α známe.',
        'Kontrolujeme tři věci: rovnici uvnitř tyče, počáteční profil a teplotu na obou koncích.',
        'V notebooku ukaž součet těchto tří ztrát a krok optimalizátoru.',
        'V grafu jsou plné čáry výsledek sítě a čárkované referenční řešení. Sledujeme, jak dobře se překrývají.',
        'Referenční řešení při trénování nepoužíváme. Slouží ke kontrole výsledku.',
    ],
    8: [
        'Teď úlohu obrátíme. Rovnici známe, ale neznáme parametr q.',
        'Známe hodnoty na obou koncích a jedno měření uprostřed: potenciál 1,625.',
        'Pro jednoduchost místo sítě použijeme parabolu ax²+bx+c.',
        'Optimalizátor současně hledá a, b, c i fyzikální parametr q.',
        'Ztráta má tři části: chybu rovnice, chybu měření a chybu okrajů.',
        'Měření uprostřed je důležité. Bez něj bychom neznámé q z této úlohy jednoznačně neurčili.',
        'Správný výsledek je q=1. V kódu ukaž, že q je zahrnuté v optimalizátoru.',
    ],
    9: [
        'Kde máme měřit, abychom dobře určili q?',
        'Citlivost říká, jak moc se změní potenciál, když trochu změníme q.',
        'Tady používáme známý analytický vzorec pro potenciál. Proto je ve vzorci f bez θ.',
        'V kódu derivujeme podle q, ne podle polohy.',
        'Na koncích je citlivost nulová, protože hodnoty jsou pevně předepsané. Největší je uprostřed.',
        'Pro tento jeden parametr a stejný šum je proto nejlepší měřit uprostřed.',
        'Pokud padne dotaz na síť: předchozí model má q jen v rovnici, ne ve svém výstupu. Jeho citlivost proto nedostaneme pouhým zavoláním autogradu podle q. Jednou možností je naučit síť se vstupy x i q.',
    ],
    10: [
        'Stejně se můžeme ptát, kdy máme měřit.',
        'Tady máme jednoduchý analytický průběh chladnutí. Nejde o celý předchozí příklad s dvoumetrovou tyčí.',
        'Parametr q v obecném vzorci je tady tepelná difuzivita α. Poloha senzoru zůstává pevná.',
        'Počítáme, jak změna α ovlivní teplotu. Vyšší α znamená rychlejší chladnutí, proto je citlivost záporná.',
        'Pro výběr času nás zajímá její absolutní hodnota.',
        'Na začátku je citlivost nulová. Po dlouhé době je také malá, protože tyč už téměř vychladla.',
        'Největší citlivost vychází asi ve 162 sekundách. Kód na své časové mřížce vybere 164 sekund.',
        'V okolí tohoto času má při stejném šumu smysl měřit hustěji.',
    ],
    11: [
        'Nestačí sledovat jen celkovou ztrátu. Sledujeme zvlášť fyziku, data a okraje.',
        'V grafu se zlepšují všechny tři části. Svislá osa je logaritmická, takže vidíme i velmi malé chyby.',
        'Malá chyba v trénovacích bodech ale ještě nezaručuje dobrý výsledek mezi nimi.',
        'Proto hotové řešení zkontrolujeme také v dalších bodech, bez nového trénování.',
        'Když někde chyba zůstává velká, změníme jednu věc, například přidáme trénovací body, a výsledek znovu porovnáme.',
        'U neznámého parametru také opakujeme trénování s různými počátečními parametry a sledujeme, jestli vychází podobně.',
        'V této ukázce s parabolou je druhá derivace konstantní. Kontrola mezi body má větší význam u obecné neuronové sítě.',
    ],
    12: [
        'Teď přejdeme k praktické úloze: rekonstrukci potenciálu z měření Kelvinovou sondovou mikroskopií.',
        'Cílem je odhadnout potenciál také v místech, kde jsme neměřili.',
        'Otevři notebook PINN_KPFM_Colab.ipynb uvedený na slidu.',
        'Nejdřív projděte zadání a měření. Potom určete, jaká rovnice a jaké podmínky se budou při učení kontrolovat.',
        'Nakonec porovnejte předpověď s kontrolními daty, která nebyla použita při trénování, pokud jsou v notebooku připravená.',
    ],
}

root = Path('F:/PINN')
path = next(p for p in root.glob('*.pptx') if not p.name.startswith('~$'))
source = path.read_bytes()
review = root / 'data/presentation_review'
backup = review / 'before_replacing_speaker_notes.pptx'
if not backup.exists():
    backup.write_bytes(source)
patch = {}
ns = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
with zipfile.ZipFile(io.BytesIO(source)) as deck:
    for i, paragraphs in notes.items():
        name = f'ppt/notesSlides/notesSlide{i}.xml'
        xml = deck.read(name).decode('utf-8')
        matches = [m for m in re.finditer(r'<p:sp>.*?</p:sp>', xml, re.S)
                   if re.search(r'<p:ph\b[^>]*\btype="body"', m.group())]
        assert len(matches) == 1, i
        match = matches[0]
        paras = ''.join('<a:p><a:r><a:rPr lang="cs-CZ"/><a:t>' + escape(t) +
                        '</a:t></a:r><a:endParaRPr lang="cs-CZ"/></a:p>' for t in paragraphs)
        body = '<p:txBody><a:bodyPr/><a:lstStyle/>' + paras + '</p:txBody>'
        shape, count = re.subn(r'<p:txBody>.*?</p:txBody>', lambda _: body, match.group(), flags=re.S)
        assert count == 1
        patched = xml[:match.start()] + shape + xml[match.end():]
        ET.fromstring(patched)
        patch[name] = patched.encode('utf-8')
    result = io.BytesIO()
    with zipfile.ZipFile(result, 'w', compression=zipfile.ZIP_DEFLATED) as output:
        for info in deck.infolist():
            output.writestr(copy.copy(info), patch.get(info.filename, deck.read(info.filename)))
    with zipfile.ZipFile(io.BytesIO(result.getvalue())) as check:
        assert check.testzip() is None
        for name in deck.namelist():
            if name not in patch:
                assert deck.read(name) == check.read(name), name
        for i, paragraphs in notes.items():
            parsed = ET.fromstring(check.read(f'ppt/notesSlides/notesSlide{i}.xml'))
            found = []
            for shape in parsed.findall('.//p:sp', ns):
                ph = shape.find('p:nvSpPr/p:nvPr/p:ph', ns)
                if ph is not None and ph.get('type') == 'body':
                    found = [''.join(p.itertext()) for p in shape.findall('p:txBody/a:p', ns)]
            assert found == paragraphs, i
    assert path.read_bytes() == source, 'Source changed while preparing notes'
    path.write_bytes(result.getvalue())
(review/'speaker_notes_expected.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
print('Replaced speaker notes on all 12 slides. All slide content, layouts, images and other package parts unchanged.')
