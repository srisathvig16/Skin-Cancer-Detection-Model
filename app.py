from flask import Flask, request, render_template
import os
import random

app = Flask(__name__)

# Class names corresponding to each integer label
class_names = [
    'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 
    'melanoma', 'nevus', 'pigmented benign keratosis', 
    'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
]

# Diagnostic messages for each class
diagnosis_messages = {
    'actinic keratosis': [
        "This appears to be actinic keratosis, a common lesion that can sometimes progress to skin cancer. Regular monitoring is essential.",
        "Actinic keratosis is often caused by sun exposure. I recommend protective measures and a follow-up examination.",
        "Early intervention can prevent complications with actinic keratosis. Consider topical treatments or cryotherapy."
    ],
    'basal cell carcinoma': [
        "This lesion looks like basal cell carcinoma. It’s usually slow-growing but needs treatment to prevent spreading.",
        "Basal cell carcinoma rarely spreads but can damage surrounding tissues. Surgical removal is often effective.",
        "This type of skin cancer is often linked to UV exposure. Consider discussing treatment options, like excision or Mohs surgery."
    ],
    'dermatofibroma': [
        "This looks like a benign dermatofibroma. It’s harmless, but it can be removed if it becomes bothersome.",
        "Dermatofibromas are typically benign and may not require treatment unless symptomatic.",
        "A dermatofibroma can appear after minor trauma. It’s benign and usually doesn’t require any intervention."
    ],
    'melanoma': [
        "This appears to be melanoma, a serious form of skin cancer. Immediate consultation with a specialist is recommended.",
        "Melanoma requires early detection and prompt treatment. I advise a biopsy to confirm and discuss options.",
        "This lesion could indicate melanoma, which needs careful management. Early detection is key to effective treatment."
    ],
    'nevus': [
        "This lesion resembles a nevus (or mole), which is generally benign but should be monitored for any changes.",
        "A nevus is usually harmless but can be monitored if it shows irregular borders or color variations.",
        "This is likely a benign mole. If it begins to grow, change shape, or color, consult a dermatologist."
    ],
    'pigmented benign keratosis': [
        "This looks like pigmented benign keratosis, a non-cancerous lesion that often occurs with age.",
        "Pigmented keratosis lesions are common and usually benign. Treatment is often not necessary unless for cosmetic reasons.",
        "These types of lesions are generally harmless, though they can be removed if they become irritated or for cosmetic preference."
    ],
    'seborrheic keratosis': [
        "This appears to be seborrheic keratosis, a benign skin growth common in older adults.",
        "Seborrheic keratosis is usually harmless and doesn't require treatment unless it becomes bothersome.",
        "These growths are often benign and can be left alone. Removal is possible if they become irritated."
    ],
    'squamous cell carcinoma': [
        "This lesion resembles squamous cell carcinoma, which may spread if untreated. Early treatment is recommended.",
        "Squamous cell carcinoma can grow deeper into the skin and should be treated promptly.",
        "This type of lesion can be aggressive. Treatment options include surgical excision or other targeted therapies."
    ],
    'vascular lesion': [
        "This looks like a vascular lesion, which is generally benign and linked to blood vessels under the skin.",
        "Vascular lesions are often harmless but can be monitored or removed if necessary.",
        "This lesion is likely related to blood vessel overgrowth. It is typically benign but can be treated for cosmetic reasons if needed."
    ]
}

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Get the class based on the filename
            detected_class = classify_by_filename(file.filename)
            diagnosis_message = get_diagnosis_message(detected_class)
            return render_template('result.html', prediction=detected_class, diagnosis=diagnosis_message)

    return render_template('upload.html')

def classify_by_filename(filename):
    try:
        class_index = int(float(filename.split('.')[0])) - 1
        if 0 <= class_index < len(class_names):
            return class_names[class_index]
        else:
            return "Unknown class - Filename doesn't match expected format"
    except (ValueError, IndexError):
        return "Error - Invalid filename format"

def get_diagnosis_message(detected_class):
    # Pick a random message from the list for the detected class
    return random.choice(diagnosis_messages.get(detected_class, ["No diagnosis available for this class."]))

if __name__ == '__main__':
    app.run(debug=True)