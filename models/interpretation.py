def request_interpretation(llm, payload):
    # An unlabelled landmark trajectory has no reliable semantic meaning.
    # Only send known gesture labels; never ask an LLM to invent an interpretation.
    labels = [e['gesture'] for e in payload['events']]
    if not labels:
        return 'No recognised gesture labels; sequence saved for training.'
    prompt = (
        'These are MediaPipe gesture categories, not verified user-defined commands: '
        + ', '.join(labels)
        + '. Describe only what can be established. If their meaning is unknown, '
        'say that the user must teach the meaning using ASL fingerspelling. '
        'Do not claim to have performed an action.'
    )
    return llm.chat(prompt)
