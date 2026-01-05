for pdf in ./papers/naacl/naacl2025-sample-pdf/*.pdf; do
  # Derive the XML filename from the PDF filename (e.g., document.pdf -> document.xml)
  xml="${pdf%.pdf}.xml"
  
  # Check if the XML file already exists
  if [ -f "$xml" ]; then
    echo "Skipping '$pdf' as '$xml' already exists."
    continue
  fi

  filename=$(basename "$pdf" .pdf)
  curl -v --form input=@"$pdf" \
    http://localhost:8070/api/processFulltextDocument > "./papers/naacl/naacl-2025/$filename.xml"

done