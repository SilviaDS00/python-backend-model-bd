from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS  # Agrega la importación de CORS
import joblib

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///respuestas.db'
db = SQLAlchemy(app)

CORS(app)

class Respuesta(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pregunta = db.Column(db.String(255))
    respuesta = db.Column(db.Integer)

# Carga el modelo desde el archivo .pkl
modelo = joblib.load('model/entrenamiento_estudiantes.pkl')

@app.route('/submit-cuestionario', methods=['POST'])
def submit_cuestionario():
    data = request.get_json()

    respuestas_usuario = [respuesta for _, respuesta in zip(data['columnas'], data['respuestas'])]
    # Realiza la predicción con el modelo
    prediccion = modelo.predict([respuestas_usuario])

    for pregunta, respuesta in zip(data['columnas'], data['respuestas']):
        nueva_respuesta = Respuesta(pregunta=pregunta, respuesta=respuesta)
        db.session.add(nueva_respuesta)

    db.session.commit()

    return jsonify({'mensaje': 'Respuestas almacenadas correctamente', 'prediccion': prediccion.tolist()})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False)

