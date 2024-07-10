from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import DataRequired

class SuntracerForm(FlaskForm):
    temperatura = FloatField('Temperatura', validators=[DataRequired(message='El valor añadido no es válido')])
    submit = SubmitField('Predecir')