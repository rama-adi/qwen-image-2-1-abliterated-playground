"""Ordered reference roles shared by prompt assembly and image resolution."""
import re

ROLES = {
    'style': 'Borrow the visual style: linework, brushwork, shading, texture, and rendering technique.',
    'pose': 'Borrow the body pose, gesture, and orientation; keep identity and clothing from the scene instructions.',
    'identity': 'Preserve the referenced character or subject identity and distinguishing facial/physical features.',
    'clothing': 'Borrow the clothing, accessories, and outfit details; keep the intended subject identity.',
    'composition': 'Borrow the framing, camera angle, subject placement, and spatial layout.',
    'background': 'Borrow the environment and background details, without copying unrelated foreground subjects.',
    'object': 'Preserve the referenced object or product design, shape, and distinguishing details.',
    'lighting': 'Borrow the lighting direction, contrast, color palette, and mood.',
}


def selected_references(data):
    refs = data.get('references', [])
    if not isinstance(refs, list) or len(refs) > 4:
        raise ValueError('Use at most 4 reference images.')
    if refs and any(data.get(key) for key in ('reference', 'reference_id', 'source_history_id')):
        raise ValueError('Choose reference cards or a legacy image input, not both.')
    result = []
    for ref in refs:
        if not isinstance(ref, dict) or not isinstance(ref.get('id'), str) or not re.fullmatch(r'[a-f0-9]{32}', ref.get('id', '')):
            raise ValueError('Invalid reference image ID.')
        role = ref.get('role', 'style')
        source = ref.get('source', 'reference')
        note = str(ref.get('note') or '').strip()
        if not isinstance(role, str) or role not in ROLES or not isinstance(source, str) or source not in ('reference', 'history') or len(note) > 500:
            raise ValueError('Invalid reference role, source, or note (maximum 500 characters).')
        result.append({'id':ref['id'], 'source':source, 'role':role, 'note':note})
    return result


def reference_instructions(data):
    refs = selected_references(data)
    if not refs: return ''
    lines = ['Reference images are numbered in the order supplied.']
    for i, ref in enumerate(refs, 1):
        instruction = 'This is the source image to edit. Preserve details not requested to change.' if i == 1 and data.get('input_mode') == 'edit' else ROLES[ref['role']]
        lines.append(f'Image {i}: {instruction}' + (f" Specific instruction: {ref['note']}" if ref['note'] else ''))
    lines.append('Use each reference for its assigned purpose. Follow the scene instructions for everything else.')
    return '\n'.join(lines)
