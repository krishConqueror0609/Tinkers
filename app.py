from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from keras.models import load_model
import numpy as np
from keras.utils import get_custom_objects
from keras.layers import Layer
import tensorflow as tf

# Define the Maxout layer
class Maxout(Layer):
    def __init__(self, num_pieces, units, **kwargs):
        self.num_pieces = num_pieces
        self.units = units
        super(Maxout, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.input_dim, self.units * self.num_pieces),
            initializer='glorot_uniform'
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units * self.num_pieces,),
            initializer='zeros'
        )
        super(Maxout, self).build(input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.kernel) + self.bias
        z = tf.reshape(z, [-1, self.units, self.num_pieces])
        return tf.reduce_max(z, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(Maxout, self).get_config()
        config.update({
            "num_pieces": self.num_pieces,
            "units": self.units
        })
        return config

# Register the updated Maxout layer
get_custom_objects().update({'Maxout': Maxout})

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = '12345'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Database setup
db = SQLAlchemy(app)

# User Authentication setup
login_manager = LoginManager()
login_manager.init_app(app)

# Model setup - Load once at startup
model_path = '/Users/kavya/Downloads/gaip/mlp_model.h5'  # Path to your trained model file
model = load_model(model_path)
model.compile(optimizer='adam', loss='binary_crossentropy')

# User class for database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

# Prediction class to store user predictions
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result = db.Column(db.String(150), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Required features for prediction
REQUIRED_FEATURES = ['BehavioralProblems', 'MemoryComplaints', 'FunctionalAssessment', 'ADL', 'MMSE']

@app.route('/')
def home():
    if current_user.is_authenticated:
        return render_template('index.html', user=current_user)
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Extract form data
            form_data = request.form
            input_values = []

            # Process the form data
            for field in REQUIRED_FEATURES:
                value = form_data.get(field)
                if value is None or value.strip() == "":
                    flash(f"Error: Missing value for {field}", 'error')
                    return redirect(url_for('home'))
                
                # Convert values to float and ensure they are in the range [0,1]
                input_values.append(float(value))

            input_data = np.array([input_values])
            prediction = model.predict(input_data)

            # Save prediction to database
            prediction_text = "Positive" if prediction[0][0] >= 0.75 else "Negative"
            new_prediction = Prediction(result=prediction_text, user_id=current_user.id)
            db.session.add(new_prediction)
            db.session.commit()

            flash(f"Prediction result: {prediction_text}", 'success')

        except Exception as e:
            flash(f"Error: {str(e)}", 'error')
            prediction_text = None  # Reset prediction in case of error

    return render_template('predict.html', prediction_text=prediction_text)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.password == password:  # Simple password check (improve with hashing)
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user:
            flash('Username already exists', 'error')
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/prediction_history')
@login_required
def prediction_history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).all()
    return render_template('prediction_history.html', predictions=predictions)

if __name__ == '__main__':
    with app.app_context():  # Ensure the app context is pushed
        db.create_all()  # Ensure database tables are created
    app.run(debug=True)

