# Epic 5 – Flask Web Application

## Objective

The objective of this epic is to build a web application that allows users to predict flood occurrence using the trained machine learning model developed in Epic 4.

---

## Technologies Used

- Python
- Flask
- HTML
- CSS
- Joblib
- Pandas

---

## Project Files

```
Epic_5_Flask_Web_Application/
│
├── app.py
├── model/
│   └── best_model.pkl
├── templates/
├── static/
├── dataset/
└── output/
```

---

## Features

- User-friendly web interface.
- Accepts environmental parameters as input.
- Uses the trained machine learning model.
- Displays prediction result:
  - Flood
  - No Flood

---

## How to Run

1. Open Terminal.

2. Navigate to the project folder.

```bash
cd Epic_5_Flask_Web_Application
```

3. Run the application.

```bash
python3 app.py
```

4. Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Output

The application predicts flood risk based on the user inputs and displays the prediction on the webpage.

---

## Conclusion

This Flask application successfully integrates the trained machine learning model with a web interface, allowing users to perform flood prediction interactively.