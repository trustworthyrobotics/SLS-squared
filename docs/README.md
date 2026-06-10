# Local GitHub Pages Preview

This project page is plain static HTML, so there is no compile step. To preview it locally from the repository root:

```bash
python3 -m http.server 8000 --directory docs
```

Then open:

```text
http://localhost:8000
```

If port 8000 is busy, choose another port:

```bash
python3 -m http.server 8080 --directory docs
```
