def make_pdf(*pages: str) -> bytes:
    """Build a minimal valid PDF with one line of Helvetica text per page."""
    count = len(pages)
    font_id = 3 + 2 * count
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{3 + 2 * page_index} 0 R" for page_index in range(count)), count
        ),
    ]
    for page_index, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET"
        contents_id = 4 + 2 * page_index
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 200] /Contents {contents_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    output += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    output += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(output)
