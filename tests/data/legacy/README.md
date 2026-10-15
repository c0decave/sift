# Real legacy Office fixtures

`book.xls`, `letter.doc` and `plain.doc` were written by LibreOffice (an
independent writer) from the flat-ODF documents `make_inputs.py` generates:

    python3 make_inputs.py DIR          # writes book.fods, letter.fodt, plain.fodt
    soffice --headless --convert-to 'xls:MS Excel 97' book.fods
    soffice --headless --convert-to 'doc:MS Word 97' letter.fodt plain.fodt

(`xmlns:of` must be declared, or LibreOffice reads every `of:=` formula as an
error.) LibreOffice's own reading of the same files, as CSV and text, is what
`tests/test_legacy_office.py` compares against.
