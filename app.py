from flask import Flask, render_template, request, redirect, send_file
import os
from ultralytics import YOLO
import cv2
import torch
from werkzeug.utils import secure_filename
import easyocr
from fpdf import FPDF

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['STATIC_FOLDER'] = 'static/'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Load the trained YOLO model
model = YOLO('./models/best_model.pt')

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save uploaded image
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform detection
            results = model(filepath)
            detections = results[0].boxes.xyxy.cpu().numpy().astype(int)
            img = cv2.imread(filepath)

            # Draw bounding boxes on the image
            for x1, y1, x2, y2 in detections:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Save the result image
            result_img_path = os.path.join(app.config['STATIC_FOLDER'], 'result.jpg')
            cv2.imwrite(result_img_path, img)

            # Prepare for PDF generation
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Page 1: Image with bounding boxes
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Detection Results', ln=True, align='C')
            pdf.ln(10)
            pdf.set_font('Arial', size=12)
            pdf.cell(0, 10, 'Original Image with Bounding Boxes:', ln=True)
            pdf.image(result_img_path, x=10, y=pdf.get_y() + 10, w=190)

            # Page 2: Text details
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Detected Objects Details', ln=True, align='C')
            pdf.ln(10)

            # Multiline text handling function
            def multiline_cell(pdf, text, max_width=120):
                """
                Custom function to handle multiline text in PDF cells
                """
                # Split text into words
                words = text.split()
                lines = []
                current_line = []

                # Break text into lines that fit within max_width
                for word in words:
                    # Test line width
                    test_line = ' '.join(current_line + [word])
                    pdf.set_font('Arial', size=12)
                    line_width = pdf.get_string_width(test_line)
                    
                    if line_width <= max_width:
                        current_line.append(word)
                    else:
                        # Add current line and start new line
                        lines.append(' '.join(current_line))
                        current_line = [word]
                
                # Add last line
                if current_line:
                    lines.append(' '.join(current_line))
                
                return lines

            # Add table-like layout for detections
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(20, 10, 'No.', 1)
            pdf.cell(50, 10, 'Bounding Box', 1)
            pdf.cell(120, 10, 'Extracted Text', 1)
            pdf.ln()

            text_results = []
            for idx, (x1, y1, x2, y2) in enumerate(detections):
                # Crop and process each detection
                cropped_img = img[y1:y2, x1:x2]
                cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                # Recognize text using EasyOCR
                text = reader.readtext(cropped_img_rgb, detail=0)
                recognized_text = ' '.join(text)
                text_results.append({'bbox': (x1, y1, x2, y2), 'text': recognized_text})

                # Add to PDF with multiline support
                pdf.set_font('Arial', size=12)
                
                # Number cell
                pdf.cell(20, 10, str(idx + 1), 1)
                
                # Bounding box cell
                pdf.cell(50, 10, f"({x1},{y1}) - ({x2},{y2})", 1)
                
                # Multiline text cell
                text_lines = multiline_cell(pdf, recognized_text)
                
                # Determine max lines for consistent cell height
                max_lines = max(1, len(text_lines))
                
                # Initial text cell
                pdf.cell(120, 10, text_lines[0] if text_lines else '', 1)
                pdf.ln()
                
                # Add additional lines if exist
                for line in text_lines[1:]:
                    pdf.cell(20, 10, '', 1)  # Empty first column
                    pdf.cell(50, 10, '', 1)  # Empty second column
                    pdf.cell(120, 10, line, 1)
                    pdf.ln()

            # Save the PDF
            pdf_path = os.path.join(app.config['STATIC_FOLDER'], 'results.pdf')
            pdf.output(pdf_path)

            return render_template('results.html', texts=text_results)
    return render_template('index.html')

@app.route('/download')
def download_pdf():
    pdf_path = os.path.join(app.config['STATIC_FOLDER'], 'results.pdf')
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)