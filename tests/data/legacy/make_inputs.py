import sys, html
out = sys.argv[1]
NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
      'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
      'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
      'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
      'xmlns:number="urn:oasis:names:tc:opendocument:xmlns:datastyle:1.0" '
      'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" '
      'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
      'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0" '
      'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" '
      'xmlns:of="urn:oasis:names:tc:opendocument:xmlns:of:1.2" office:version="1.2"')
e = html.escape
def s(v): return f'<table:table-cell office:value-type="string"><text:p>{e(v)}</text:p></table:table-cell>'
def n(v): return f'<table:table-cell office:value-type="float" office:value="{v}"><text:p>{v}</text:p></table:table-cell>'
def d(v): return f'<table:table-cell table:style-name="ce1" office:value-type="date" office:date-value="{v}"><text:p>{v}</text:p></table:table-cell>'
def b(v): return f'<table:table-cell office:value-type="boolean" office:boolean-value="{str(v).lower()}"><text:p>{str(v).upper()}</text:p></table:table-cell>'
def f(expr, v): return f'<table:table-cell table:formula="{e(expr)}" office:value-type="float" office:value="{v}"><text:p>{v}</text:p></table:table-cell>'
def sf(expr, v): return f'<table:table-cell table:formula="{e(expr)}" office:value-type="string" office:string-value="{e(v)}"><text:p>{e(v)}</text:p></table:table-cell>'
def row(*cells, hidden=False):
    a = ' table:visibility="collapse"' if hidden else ''
    return f'<table:table-row{a}>' + ''.join(cells) + '</table:table-row>'
empty = '<table:table-cell/>'
long_text = ("Lorem ipsum dolor sit amet, " * 14).strip()
people = [row(s("Name"), s("Amount"), s("Joined"), s("Active"), s("Note")),
          row(s("Müller"), n("1234.5"), d("2024-03-05"), b(True), s("Grüße € ☃")),
          row(s("O'Brien"), n("-7"), d("1999-12-31"), b(False), s(long_text)),
          row(s("Σίσυφος"), n("0.1"), d("2024-02-29"), b(True), empty),
          row(s("Total"), f("of:=SUM([.B2:.B4])", "1227.6"), f("of:=[.B2]*2", "2469"), empty, empty),
          row(s("hidden row secret"), n("1"), empty, empty, empty, hidden=True),
          row(empty, empty, empty, empty, s("far right"))]
many = [row(s(f"unique string number {i:05d} with some padding to make it long enough")) for i in range(300)]
nums = [row(n("0.1"), n("1E+21"), n("123456789012"), n("-0.000001"), n("100"), n("3.14159265358979"))]
hidden_sheet = [row(s("secret-in-hidden-sheet"), n("42"))]
sheets = (f'<table:table table:name="People">{"".join(people)}</table:table>'
          f'<table:table table:name="Zahlen">{"".join(nums)}</table:table>'
          f'<table:table table:name="Many">{"".join(many)}</table:table>'
          f'<table:table table:name="Hidden" table:style-name="tahidden">{"".join(hidden_sheet)}</table:table>')
fods = (f'<?xml version="1.0" encoding="UTF-8"?><office:document {NS} office:mimetype="application/vnd.oasis.opendocument.spreadsheet">'
        '<office:automatic-styles><number:date-style style:name="N84"><number:year number:style="long"/><number:text>-</number:text>'
        '<number:month number:style="long"/><number:text>-</number:text><number:day number:style="long"/></number:date-style>'
        '<style:style style:name="ce1" style:family="table-cell" style:data-style-name="N84"/>'
        '<style:style style:name="tahidden" style:family="table"><style:table-properties table:display="false"/></style:style>'
        f'</office:automatic-styles><office:body><office:spreadsheet>{sheets}</office:spreadsheet></office:body></office:document>')
open(f"{out}/book.fods", "w", encoding="utf-8").write(fods)

def p(t, style=None): return f'<text:p>{e(t)}</text:p>'
body = (f'<text:h text:outline-level="1">Überschrift Eins</text:h>'
        + p("First paragraph with umlauts: äöüß and a euro sign €.")
        + p("Greek: Σίσυφος and a snowman ☃ force Unicode pieces.")
        + '<table:table table:name="T1"><table:table-column table:number-columns-repeated="2"/>'
        + '<table:table-row><table:table-cell office:value-type="string"><text:p>cell-a1</text:p></table:table-cell><table:table-cell office:value-type="string"><text:p>cell-b1</text:p></table:table-cell></table:table-row>'
        + '<table:table-row><table:table-cell office:value-type="string"><text:p>cell-a2</text:p></table:table-cell><table:table-cell office:value-type="string"><text:p>cell-b2</text:p></table:table-cell></table:table-row></table:table>'
        + p("Final paragraph. ".join(["word"] * 3) + " " + "long " * 800))
fodt = (f'<?xml version="1.0" encoding="UTF-8"?><office:document {NS} office:mimetype="application/vnd.oasis.opendocument.text">'
        f'<office:body><office:text>{body}</office:text></office:body></office:document>')
open(f"{out}/letter.fodt", "w", encoding="utf-8").write(fodt)
# latin-only doc (8-bit pieces)
body2 = p("Plain ASCII and latin1 only: Grüße aus Köln.") + p("Second line.")
open(f"{out}/plain.fodt", "w", encoding="utf-8").write(
    f'<?xml version="1.0" encoding="UTF-8"?><office:document {NS} office:mimetype="application/vnd.oasis.opendocument.text">'
    f'<office:body><office:text>{body2}</office:text></office:body></office:document>')
def page(i, title, bullets):
    items = ''.join(f'<text:list-item><text:p>{e(b_)}</text:p></text:list-item>' for b_ in bullets)
    return (f'<draw:page draw:name="page{i}" draw:master-page-name="Default">'
            f'<draw:frame presentation:class="title" svg:width="20cm" svg:height="2cm" svg:x="2cm" svg:y="1cm"><draw:text-box><text:p>{e(title)}</text:p></draw:text-box></draw:frame>'
            f'<draw:frame presentation:class="outline" svg:width="20cm" svg:height="8cm" svg:x="2cm" svg:y="4cm"><draw:text-box><text:list>{items}</text:list></draw:text-box></draw:frame>'
            '</draw:page>')
pres = page(1, "Quarterly Überblick", ["first bullet", "second bullet Müller"]) + page(2, "Slide two", ["alpha", "beta"])
open(f"{out}/deck.fodp", "w", encoding="utf-8").write(
    f'<?xml version="1.0" encoding="UTF-8"?><office:document {NS} office:mimetype="application/vnd.oasis.opendocument.presentation">'
    f'<office:body><office:presentation>{pres}</office:presentation></office:body></office:document>')
