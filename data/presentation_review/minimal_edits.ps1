$ErrorActionPreference = 'Stop'
$deckPath = (Get-ChildItem -LiteralPath 'F:\PINN' -Filter '*.pptx' | Select-Object -First 1).FullName
$reviewDir = 'F:\PINN\data\presentation_review'
$backup = Join-Path $reviewDir 'before_minimal_edits.pptx'
if (-not (Test-Path -LiteralPath $backup)) { Copy-Item -LiteralPath $deckPath -Destination $backup }
$ppt = New-Object -ComObject PowerPoint.Application
$deck = $ppt.Presentations.Open($deckPath, 0, 0, 0)
function Find-TextShape($slide, $needle) {
    foreach ($shape in $slide.Shapes) {
        if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
            if ($shape.TextFrame.TextRange.Text.Contains($needle)) { return $shape }
        }
    }
    throw "Text shape not found: $needle"
}
function Add-Footnote($slide, $top, $text, $size) {
    $box = $slide.Shapes.AddTextbox(1, 51, $top, 814, 35)
    $box.Name = 'Poznámka pod čarou'
    $box.TextFrame.MarginLeft = 0
    $box.TextFrame.MarginRight = 0
    $box.TextFrame.MarginTop = 0
    $box.TextFrame.MarginBottom = 0
    $box.TextFrame.WordWrap = -1
    $box.TextFrame.TextRange.Text = $text
    $box.TextFrame.TextRange.Font.Name = 'Aptos'
    $box.TextFrame.TextRange.Font.Size = [single]$size
    $box.TextFrame.TextRange.Font.Bold = 0
    $box.TextFrame.TextRange.Font.Color.RGB = 9073236
    $box.TextFrame.TextRange.ParagraphFormat.Bullet.Visible = 0
    $box.TextFrame.TextRange.ParagraphFormat.SpaceBefore = 0
    $box.TextFrame.TextRange.ParagraphFormat.SpaceAfter = 0
    return $box
}
function Append-Notes($slide, $text) {
    foreach ($shape in $slide.NotesPage.Shapes) {
        if ($shape.Type -eq 14 -and $shape.PlaceholderFormat.Type -eq 2) {
            $existing = $shape.TextFrame.TextRange.Text.TrimEnd()
            $shape.TextFrame.TextRange.Text = $existing + "`r`r" + $text
            return
        }
    }
    throw 'Speaker notes placeholder not found'
}
try {
    $s6 = $deck.Slides.Item(6)
    $body = Find-TextShape $s6 'pro Neumannovu'
    $found = $false
    for ($i = 1; $i -le $body.TextFrame.TextRange.Paragraphs().Count; $i++) {
        $para = $body.TextFrame.TextRange.Paragraphs($i)
        if ($para.Text.Contains('pro Neumannovu')) {
            $start = $para.Start
            $count = $para.Length
            if ($para.Text.EndsWith("`r")) { $count-- }
            $replacement = $body.TextFrame.TextRange.Characters($start, $count)
            $replacement.Delete()
            $plain = $s6.Shapes.AddTextbox(1, 75, 421, 450, 30)
            $plain.Name = 'Upřesnění Neumannových podmínek'
            $plain.TextFrame.MarginLeft = 0
            $plain.TextFrame.MarginRight = 0
            $plain.TextFrame.MarginTop = 0
            $plain.TextFrame.MarginBottom = 0
            $plain.TextFrame.TextRange.Text = 'Pro Neumannovy podmínky je nutný jiný pomocný tvar.'
            $plain.TextFrame.TextRange.Font.Name = 'Aptos'
            $plain.TextFrame.TextRange.Font.Size = 15
            $plain.TextFrame.TextRange.Font.Bold = 0
            $plain.TextFrame.TextRange.Font.Italic = 0
            $plain.TextFrame.TextRange.Font.Color.RGB = 4862485
            $found = $true
            break
        }
    }
    if (-not $found) { throw 'Neumann paragraph not found' }
    $note6 = 'Pozn.: Kvadratická aktivace σ(z) = z² s afinním vstupem a výstupem umožní φ(x) = 17/8 − ½(x − 3/2)²; tato funkce splní okraje i φ″ = −1. Samotná změna aktivace obecně splnění okrajů nezaručuje.'
    $null = Add-Footnote $s6 491 $note6 10
    Append-Notes $s6 ('Uvedený pomocný tvar přesně vynucuje Dirichletovy hodnoty, nikoli Neumannovy derivace. Pro Neumannovy podmínky je potřeba jiná konstrukce; u normálové derivace záleží také na orientaci normály.' + "`r`r" + 'Ukázka přes okrajovou ztrátu pouze sestavuje loss: tanh(wx) má vždy φ(0) = 0 a nemůže dosáhnout φ(1) = 2. Vhodnější parametrizací je např. model a·σ(bx+c)+d. Pro σ(z)=z² volba a=−1/2, b=1, c=−3/2, d=17/8 dává φ(x)=1+3x/2−x²/2. Tato konkrétní funkce splňuje obě okrajové podmínky i Poissonovu rovnici. Běžná změna aktivace sama o sobě přesné splnění podmínek nevynucuje.')

    $s9 = $deck.Slides.Item(9)
    $body9 = Find-TextShape $s9 'Pro maximalizování'
    $old9 = 'Pro maximalizování informace z měření je dobré dát senzor na místa, kde je citlivost největší'
    $new9 = 'Pro jeden parametr a stejný šum měříme tam, kde je |S| největší'
    $result = $body9.TextFrame.TextRange.Replace($old9, $new9)
    if ($null -eq $result) { throw 'Sensitivity spatial text not replaced' }
    $note9 = 'Pozn.: Ukázka derivuje analytické řešení φ(x;q). Pro přímou derivaci sítě podle q musí být q jejím vstupem; model fθ(x) s učeným q pouze v reziduu tuto závislost nemá.'
    $null = Add-Footnote $s9 461 $note9 11
    Append-Notes $s9 'Citlivost je derivace rodiny řešení podle fyzikálního parametru. Ukázka používá analytický vztah. U inverzního modelu fθ(x), kde q vstupuje pouze do fyzikálního rezidua, nelze tuto citlivost získat pouhou derivací výstupu podle samostatného q při pevných vahách. Možností je parametrická síť fθ(x,q), citlivostní rovnice nebo derivace řešení přes optimalizaci. Maximum |S| přímo pomáhá pro jeden parametr při stejném nezávislém šumu; u více parametrů je nutné zohlednit jejich rozlišitelnost.'

    $s10 = $deck.Slides.Item(10)
    $body10 = Find-TextShape $s10 'Je dobré stanovit interval'
    $old10 = 'Je dobré stanovit interval, kde je nejdůležitější provést více měření/hustší sampling'
    $new10 = 'Pro jeden parametr a stejný šum měříme hustěji v časech, kde je |S| největší'
    $result = $body10.TextFrame.TextRange.Replace($old10, $new10)
    if ($null -eq $result) { throw 'Sensitivity temporal text not replaced' }
    $note10 = 'Pozn.: Ukázka derivuje analytické řešení podle q = α při pevné poloze senzoru. Pro přímou derivaci sítě podle q musí být q jejím vstupem.'
    $null = Add-Footnote $s10 484 $note10 10.5
    Append-Notes $s10 'Ukázka používá jeden analytický mód teplotní odchylky ΔT=40 exp(−αk²t). Citlivost S=−40k²t exp(−αk²t) je záporná, takže při volbě měřicích časů porovnáváme její absolutní hodnotu. Maximum je přibližně v 162 s. Poloha senzoru je pevná a q v obecném vzorci odpovídá difuzivitě α. Přímá derivace parametrické sítě podle q vyžaduje její explicitní závislost na q.'

    $deck.SaveAs((Join-Path $reviewDir 'minimal_edits_rendered.pptx'), 24)
    foreach ($n in @(6,9,10)) {
        $deck.Slides.Item($n).Export((Join-Path $reviewDir "after_$n.png"), 'PNG', 1600, 900)
    }
    Write-Output 'Edited slides 6, 9, 10; rendered previews saved.'
} finally {
    $deck.Close()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($deck)
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt)
}
