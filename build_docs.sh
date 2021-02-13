pdoc --pdf ODElib >pdf.md
pandoc --metadata=title:"ODElib Documentation" --toc --toc-depth=4 --from=markdown+abbreviations --pdf-engine=xelatex --variable=mainfont:"DejaVu Sans" --output=ODElib_docs.pdf pdf.md
