# filter_recommender.py

def recommend_filters(color_mood, emotion, theme):
    recommendations = []

    if color_mood == 'gloomy':
        recommendations.extend(['vivid', 'warm', 'dramatic'])
    elif color_mood == 'energetic':
        recommendations.extend(['vivid', 'warm', 'cool'])
    elif color_mood == 'calm':
        recommendations.extend(['cool', 'monochrome', 'impressionist'])
    elif color_mood == 'joyful':
        recommendations.extend(['vivid', 'warm', 'nostalgic'])
    else:  # neutral
        recommendations.extend(['vivid', 'warm', 'monochrome'])

    if emotion in ['sad', 'fear', 'angry']:
        recommendations.extend(['vivid', 'warm', 'dramatic'])
        if 'cool' in recommendations:
            recommendations.remove('cool')
    elif emotion in ['happy', 'surprise']:
        recommendations.extend(['vivid', 'warm'])
    elif emotion == 'neutral':
        recommendations.extend(['monochrome', 'cool', 'impressionist'])

    if theme == 'portrait':
        recommendations.extend(['warm', 'monochrome', 'nostalgic'])
    elif theme == 'landscape':
        recommendations.extend(['vivid', 'cool', 'dramatic'])
    elif theme == 'food':
        recommendations.extend(['warm', 'vivid'])
    elif theme == 'animal':
        recommendations.extend(['warm', 'vivid'])
    elif theme == 'nature':
        recommendations.extend(['vivid', 'cool', 'warm'])

    # Remove duplicates and return top 4
    unique_recommendations = []
    for rec in recommendations:
        if rec not in unique_recommendations:
            unique_recommendations.append(rec)
        if len(unique_recommendations) >= 4:
            break

    return unique_recommendations[:4]
