def generate_recommendations(
    hours_studied,
    attendance,
    sleep_hours,
    tutoring_sessions,
    physical_activity,
    motivation
):
    recommendations = []
    strengths = []

    # Study Hours
    if hours_studied < 15:
        recommendations.append("📚 Increase study hours to at least 20 hours per week.")
    else:
        strengths.append("📚 Good study routine.")

    # Attendance
    if attendance < 85:
        recommendations.append("🏫 Improve attendance to above 85%.")
    else:
        strengths.append("🏫 Excellent attendance.")

    # Sleep
    if sleep_hours < 7:
        recommendations.append("😴 Sleep 7–8 hours daily for better concentration.")
    else:
        strengths.append("😴 Healthy sleeping habits.")

    # Tutoring
    if tutoring_sessions < 2:
        recommendations.append("👨‍🏫 Consider attending more tutoring sessions.")
    else:
        strengths.append("👨‍🏫 Good academic support.")

    # Physical Activity
    if physical_activity < 3:
        recommendations.append("🏃 Increase physical activity for better overall performance.")
    else:
        strengths.append("🏃 Healthy physical activity level.")

    # Motivation
    if motivation == "Low":
        recommendations.append("🔥 Work on improving motivation through goals and regular study plans.")
    elif motivation == "Medium":
        recommendations.append("💡 Increasing motivation can further improve performance.")
    else:
        strengths.append("🔥 Highly motivated learner.")

    return strengths, recommendations