"""Render the fact-checked report source and recorded metrics into a PDF."""

from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


source = json.loads(Path('report-source.json').read_text())
results = json.loads(Path('results/summary.json').read_text())
styles = {
    'title': ParagraphStyle('title', fontName='Times-Bold', fontSize=24, leading=27, spaceAfter=5),
    'subtitle': ParagraphStyle('subtitle', fontName='Helvetica', fontSize=11, leading=14, textColor=colors.HexColor('#325c60'), spaceAfter=8),
    'author': ParagraphStyle('author', fontName='Helvetica', fontSize=9, leading=12, spaceAfter=16),
    'heading': ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=11, leading=14, spaceBefore=12, spaceAfter=6, keepWithNext=True),
    'body': ParagraphStyle('body', fontName='Times-Roman', fontSize=10.5, leading=13.4, spaceAfter=7, allowWidows=False, allowOrphans=False),
    'caption': ParagraphStyle('caption', fontName='Helvetica', fontSize=8.4, leading=10.6, spaceAfter=7),
    'reference': ParagraphStyle('reference', fontName='Times-Roman', fontSize=9, leading=11.2, spaceAfter=5),
    'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=9, leading=12, alignment=TA_LEFT),
}


def paragraph(text, style='body'):
    return Paragraph(escape(text), styles[style])


def table(data, widths, numeric_start=2):
    numeric_style = ParagraphStyle('numeric', parent=styles['cell'], alignment=TA_RIGHT)
    cells = [
        [Paragraph(escape(str(cell)), numeric_style if column >= numeric_start else styles['cell'])
         for column, cell in enumerate(row)]
        for row in data
    ]
    result = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    result.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor('#648184')),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eef3f3')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
    ]))
    return result


def find(architecture, condition, split):
    return next(row for row in results['summaries'] if row['architecture'] == architecture and row['condition'] == condition and row['split'] == split)


def metrics(row):
    return f"{row['accuracy_mean'] * 100:.1f} / {row['f1_mean'] * 100:.1f}"


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#607175'))
    canvas.drawString(72, 32, 'Adam Torres | CMPSC 448 | Individual midterm project')
    canvas.drawRightString(540, 32, str(document.page))
    canvas.restoreState()


story = [paragraph(source['title'], 'title'), paragraph(source['subtitle'], 'subtitle'),
         paragraph('Adam Torres | CMPSC 448 | October 11, 2026', 'author')]
for section in source['sections']:
    story.append(paragraph(section['heading'], 'heading'))
    for text in section['paragraphs']:
        story.append(paragraph(text))
    if section.get('table') == 'main':
        data = [['Model', 'Input', 'General: acc. / F1', 'Gaming: acc. / F1']]
        for architecture in ('cnn', 'lstm'):
            for condition, label in [('input', 'Prompt'), ('output', 'Response'), ('both', 'Both')]:
                data.append([architecture.upper(), label, metrics(find(architecture, condition, 'test')), metrics(find(architecture, condition, 'game'))])
        story.append(table(data, [53, 75, 169, 171]))
        story.append(Spacer(1, 6))
        story.append(paragraph('Table 1. Accuracy and macro-F1, in percent, averaged over seeds 11, 29, and 47. General n = 396; gaming n = 93. Macro-F1 gives each generator equal weight.', 'caption'))
    if section.get('figure'):
        story.append(KeepTogether([
            Image('results/accuracy.png', width=468, height=182),
            paragraph('Figure 1. Mean accuracy. Error bars show one standard deviation across three training seeds; the dotted line marks 33.3% chance. These bars do not measure uncertainty over new prompts.', 'caption'),
        ]))
    if section.get('table') == 'ablation':
        data = [['Response condition', 'CNN gaming accuracy', 'LSTM gaming accuracy']]
        for condition, label in [('output', 'Original'), ('stripped_test_only', 'Stripped only at test'), ('normalized', 'Stripped and retrained')]:
            data.append([label, f"{find('cnn', condition, 'game')['accuracy_mean'] * 100:.1f}%", f"{find('lstm', condition, 'game')['accuracy_mean'] * 100:.1f}%"])
        story.append(table(data, [190, 139, 139], numeric_start=1))
        story.append(Spacer(1, 6))
        story.append(paragraph('Table 2. The ablation preserves the original response excerpt. It removes common Markdown markers and whitespace structure, not all possible style cues.', 'caption'))

story.append(paragraph('References', 'heading'))
for reference in source['references']:
    text = escape(reference['text']) + ' <link href="' + escape(reference['url'], quote=True) + '" color="#325c60">' + escape(reference['label']) + '</link>.'
    story.append(Paragraph(text, styles['reference']))

story.append(Spacer(1, 5))
story.append(paragraph(source['assistance'], 'caption'))
SimpleDocTemplate(
    'report.pdf', pagesize=(8.5 * inch, 11 * inch),
    rightMargin=72, leftMargin=72, topMargin=48, bottomMargin=48,
    title=source['title'], author='Adam Torres',
).build(story, onFirstPage=footer, onLaterPages=footer)
print('Wrote report.pdf')
