from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from docx import Document

# Load the dataset from the .docx file
doc = Document("extracted_comments_en.docx")

# Extract text from the document
full_text = []
for para in doc.paragraphs:
    full_text.append(para.text)

# Join all text into one string
text = ' '.join(full_text)

# Define a list of words to exclude
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(['said', 'good', 'will', 'of', 'and', 'the',
                         'dont', 'is', 'when', 'reason', 'in', 'it', 'a',
                         'If', 'you', 'at', 'for', 'don', 'all', 'as',
                         'but', 'this', 'to', 'be', 'that', 'on', 'I',
                         'he', 'with', 'are', 'have', 'there', 's', 'one', 'game',
                         'want', 'even', 'post', 'people', 'many', 'first', 'really',
                         'know', 'think', 'still', 't'])  # Replace with your list of words to exclude

# Generate a word cloud image
wordcloud = WordCloud(width = 800, height = 400,
                      background_color ='white',
                      stopwords = custom_stopwords,
                      min_font_size = 10).generate(text)

# Display the generated image:
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)

# Save the word cloud image to file
wordcloud.to_file("word_cloud.png")

plt.show()
