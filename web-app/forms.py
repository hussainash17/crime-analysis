from flask_wtf import FlaskForm
from wtforms import TextField, BooleanField, TextAreaField, SubmitField

class ContactForm(FlaskForm):
    country = TextField("Country")
    region = TextField("Region")
    year = TextField("Year")
    latitude = TextField("Latitude")
    longitude = TextField("Longitude")
    targetType = TextField("TargetType")
    attackType = TextField("AttackType")
    weaponType = TextField("WeaponType")
    weaponSubType = TextField("weaponSubType")
    suicide = TextField("Suicide")
    nkill = TextField('Nkill')
    nwonded = TextField("Nwonded")
    message = TextAreaField("Message")
    submit = SubmitField("Send")

