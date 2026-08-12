`flowchart.png` is the manuscript's Figure 1, rendered from `figure/flowchart.pdf`:

    python -c "import pypdfium2 as p; p.PdfDocument('flowchart.pdf')[0].render(scale=2.2).to_pil().save('flowchart.png')"
