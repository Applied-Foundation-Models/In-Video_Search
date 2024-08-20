import os

from flask import Blueprint, jsonify, request
from models import Video, db
from werkzeug.utils import secure_filename

# Define the Blueprint for the main routes
main = Blueprint("main", __name__)


# Define the path where videos will be saved
VIDEO_UPLOAD_FOLDER = os.path.join(os.getcwd(), "data/raw")


@main.route("/upload_video", methods=["POST"])
def upload_video():
    # Ensure the upload folder exists
    os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)

    # Get the form data
    title = request.form.get("title")
    user_id = request.form.get("user_id")

    # Get the video file from the request
    video_file = request.files["video"]

    if not video_file:
        return jsonify({"error": "No video file provided"}), 400

    # Secure the file name and save it
    filename = secure_filename(video_file.filename)
    file_path = os.path.join(VIDEO_UPLOAD_FOLDER, filename)
    video_file.save(file_path)

    # Generate the storage URL (could be a relative path)
    storage_url = f"/data/raw/{filename}"

    # Create a new Video entry in the database
    new_video = Video(title=title, storage_url=storage_url, user_id=user_id)
    db.session.add(new_video)
    db.session.commit()

    return (
        jsonify({"message": "Video uploaded successfully", "video_id": new_video.id}),
        201,
    )


@main.route("/videos/<int:video_id>", methods=["GET"])
def get_video(video_id):
    video = Video.query.get(video_id)
    if not video:
        return jsonify({"error": "Video not found"}), 404

    return (
        jsonify(
            {
                "id": video.id,
                "title": video.title,
                "storage_url": video.storage_url,  # Ensure this is correct
                "upload_date": video.upload_date.isoformat(),
            }
        ),
        200,
    )


@main.route("/videos", methods=["GET"])
def list_videos():
    videos = Video.query.all()
    return jsonify([{"id": video.id, "title": video.title} for video in videos])
