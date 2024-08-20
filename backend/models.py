from datetime import datetime

from app import db

# •	Next.js Frontend: Available at http://localhost:3000
# •	Flask Backend: Available at http://localhost:4000
# •	PostgreSQL Database: Accessible on localhost:5432 (though typically only your backend will directly interact with it)


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    videos = db.relationship("Video", backref="user", lazy=True)

    def __repr__(self):
        return f"<User {self.username}>"


class Video(db.Model):
    __tablename__ = "videos"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    storage_url = db.Column(db.String(255), nullable=False)
    keyframes = db.relationship("Keyframe", backref="video", lazy=True)

    def __repr__(self):
        return f"<Video {self.title}>"


class Keyframe(db.Model):
    __tablename__ = "keyframes"

    id = db.Column(db.Integer, primary_key=True)
    img_path = db.Column(db.String(255), nullable=False)
    timestamps = db.Column(db.Float, nullable=False)
    transcription = db.Column(db.Text, nullable=True)
    ocr_extracted_text = db.Column(db.Text, nullable=True)
    llava_result = db.Column(db.Text, nullable=True)
    keyframe_embedding = db.Column(
        db.ARRAY(db.Float), nullable=False
    )  # Assuming the embedding is a list of floats
    video_id = db.Column(db.Integer, db.ForeignKey("videos.id"), nullable=False)

    def __repr__(self):
        return f"<Keyframe {self.img_path} at {self.timestamps}s>"
