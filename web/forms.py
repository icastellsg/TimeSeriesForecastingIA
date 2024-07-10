from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import DataRequired

class SuntracerForm(FlaskForm):
    temperatura = FloatField('Temperatura', validators=[DataRequired()])
    submit = SubmitField('Predecir')

DataRequired.message = 'Please enter a valid number'