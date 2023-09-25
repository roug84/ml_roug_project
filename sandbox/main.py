# from flask import Flask
# import pandas as pd
#
#
#
# app = Flask(__name__)

#
# @app.route('/predict', methods=['POST'])
# def predict():
#     json_ = request.json
#     query_df = pd.DataFrame(json_)
#     query = pd.get_dummies(query_df)
#
#     classifier = joblib.load('classifier.pkl')
#     prediction = classifier.predict(query)
#     return jsonify({'prediction': list(prediction)})

"""
Flask running blueprints
"""
from flask import Flask
import os
from dotenv import load_dotenv
from views.index import bp as index_bp
from views.upload_file import bp10 as upload_file_bp
#from app_home_sensors.views.looknotes import bp6 as looknotes_bp

load_dotenv()
app = Flask(__name__)
name = os.getenv('NAME')

# Config email to send notifications
# app.config.update(
#     DEBUG=True,
#     # email configuration
#     MAIL_SERVER='smtp.gmail.com',
#     MAIL_PORT=465,
#     MAIL_USE_SSL=True,
#     MAIL_USERNAME='diobemex@gmail.com',
#     MAIL_PASSWORD='290811roughm'
#     )
# mail = Mail(app)

# Register BPs
app.register_blueprint(index_bp)
# app.register_blueprint(createnote_bp)
# app.register_blueprint(login_bp)
# app.register_blueprint(logout_bp)
# # app.register_blueprint(register_bp)
# app.register_blueprint(sec_cam_bp)
# app.register_blueprint(strehome_bp)
# app.register_blueprint(sendmail_bp)
# app.register_blueprint(act_stream_bp)
app.register_blueprint(upload_file_bp)


# @app.route('/', methods=['GET', 'POST'])
# def login():
#     # error = None
#     # if request.method == 'POST':
#     #     if request.form['username'] != 'hector' or request.form['password'] != 'roughm':
#     #         error = 'Invalid Credentials. Please try again.'
#     #     else:
#     #         return render_template("homepagev5.html")
#     return render_template('homepagev5.html')

    #
    # ), error=error)
# def home():
#     return "Hello world"

#
if __name__ == '__main__':
    # app.secret_key = os.urandom(12)
    #app.run(debug = True)
    app.run(port='8000')