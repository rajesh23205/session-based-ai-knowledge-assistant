from pathlib import Path

from reportlab.pdfgen import canvas

pdf_path = Path("documents/machine_learning_guide.pdf")

pdf = canvas.Canvas(str(pdf_path))

pages = [
    (
        "Introduction to Machine Learning",
        [
            "Machine learning is a branch of artificial intelligence.",
            "It allows computers to learn patterns from data.",
            "Machine learning models improve their performance through experience.",
        ],
    ),
    (
        "Supervised and Unsupervised Learning",
        [
            "Supervised learning trains models using labeled examples.",
            "Classification and regression are supervised learning tasks.",
            "Unsupervised learning finds patterns in unlabeled data.",
            "Clustering is a common unsupervised learning technique.",
        ],
    ),
    (
        "Deep Learning and Applications",
        [
            "Deep learning is a subset of machine learning.",
            "It uses neural networks with multiple layers.",
            "Applications include image recognition, speech recognition, and prediction.",
        ],
    ),
]

for title, lines in pages:
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, 780, title)

    pdf.setFont("Helvetica", 12)

    y = 740
    for line in lines:
        pdf.drawString(50, y, line)
        y -= 30

    pdf.showPage()

pdf.save()

print(f"PDF created: {pdf_path}")
