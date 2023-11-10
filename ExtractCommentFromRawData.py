import pandas as pd
from docx import Document

# Load the CSV file
file_path = 'Resources/Data/WebData/Uwants-離職原因-2023-11-07_18-42-13.csv'

# Attempt to read the CSV file with different encodings commonly used for Chinese text
for encoding in ['utf-8', 'gbk', 'gb18030']:
    try:
        data = pd.read_csv(file_path, encoding=encoding)
        break
    except UnicodeDecodeError:
        continue

# Extracting all comments from the column "主要内容"
comments = data['主要内容'].tolist()

# Create a new Document
doc = Document()

# Add each comment as a new paragraph
for comment in comments:
    doc.add_paragraph(str(comment))

# Save the document
docx_file_path = 'Resources/Data/WebData/extracted_comments.docx'
doc.save(docx_file_path)
