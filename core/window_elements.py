import uiautomation as auto

def walk_control(control, indent=0, control_type=None, search_strings=None):
    matched = []
    unmatched = []
    if control is None:
        return matched, unmatched
    if control_type is None or control.ControlType == control_type:
        try:
            rect = control.BoundingRectangle
            area = (rect.right - rect.left) * (rect.bottom - rect.top)
            # Calculate center position
            center_x = (rect.left + rect.right) // 2
            center_y = (rect.top + rect.bottom) // 2
            # Create tuple with control, area, and center coordinates
            control_tuple = (control, area, (center_x, center_y))
            if search_strings and any(s.lower() in control.Name.lower() for s in search_strings):
                matched.append(control_tuple)
            else:
                unmatched.append(control_tuple)
        except Exception as e:
            print(f"{' ' * indent}Error getting properties: {e}")

    for child in control.GetChildren():
        child_matched, child_unmatched = walk_control(child, indent + 2, control_type=control_type, search_strings=search_strings)
        matched.extend(child_matched)
        unmatched.extend(child_unmatched)
    return matched, unmatched


def sort_and_categorize_rects(controls_with_rects, size_category_to_print=None):
    sorted_by_area = sorted(controls_with_rects, key=lambda x: x[1], reverse=True)
    categorized = {'Big': [], 'Medium': [], 'Small': []}

    # First categorize by size
    for control, area, center in sorted_by_area:
        if area >= 1000000:
            categorized['Big'].append((control, center))
        elif area >= 100000:
            categorized['Medium'].append((control, center))
        else:
            categorized['Small'].append((control, center))

    output = []
    for category, controls in categorized.items():
        output.append(f"{category} Elements:")
        
        # Separate zero-coordinate controls and normal controls
        zero_coords = []
        normal_coords = []
        for control, center in controls:
            if center[0] == 0 and center[1] == 0:
                zero_coords.append(control)
            elif center[0] >= 0 and center[1] >= 0:
                normal_coords.append((control, center))

        # Process normal coordinates with grouping
        position_groups = {}
        for control, center in normal_coords:
            pos_key = (center[0], center[1])
            if pos_key not in position_groups:
                position_groups[pos_key] = []
            position_groups[pos_key].append(control)

        # Process each unique position
        for center, controls_at_pos in position_groups.items():
            # Collect unique attributes
            types = set()
            class_names = set()
            automation_ids = set()
            names = set()

            for control in controls_at_pos:
                control_type_text = auto.ControlTypeNames.get(control.ControlType, f"Unknown({control.ControlType})")
                control_type_text = control_type_text.replace("Control", "").strip()
                types.add(control_type_text)
                
                if control.ClassName:
                    class_names.add(control.ClassName)
                if control.AutomationId:
                    automation_ids.add(control.AutomationId)
                if control.Name:
                    names.add(control.Name)

            # Build the combined info string
            info_parts = [
                f"Control Types: {', '.join(sorted(types)) if types else 'Unknown'}",
                f"Class Names: {', '.join(sorted(class_names)) if class_names else 'Unknown'}",
                f"Automation IDs: {', '.join(sorted(automation_ids)) if automation_ids else 'Unknown'}",
                f"Names: {', '.join(f"'{name}'" for name in sorted(names)) if names else 'Unknown'}",
                f"Center Position: x={center[0]}, y={center[1]}"
            ]
            
            control_info = "- " + "; ".join(info_parts) + "."
            output.append(control_info)

        # Process zero-coordinate controls with deduplication
        zero_coord_groups = {}
        for control in zero_coords:
            # Create a tuple of all attributes to use as a key for grouping
            control_type_text = auto.ControlTypeNames.get(control.ControlType, f"Unknown({control.ControlType})").replace("Control", "").strip()
            attrs_key = (
                control_type_text,
                control.ClassName or "Unknown",
                control.AutomationId or "Unknown",
                control.Name or "Unknown"
            )
            
            if attrs_key not in zero_coord_groups:
                zero_coord_groups[attrs_key] = 1
            else:
                zero_coord_groups[attrs_key] += 1

        # Output unique zero-coordinate controls
        for attrs, count in zero_coord_groups.items():
            control_type, class_name, automation_id, name = attrs
            info_parts = [
                f"Control Types: {control_type}",
                f"Class Names: {class_name}",
                f"Automation IDs: {automation_id}",
                f"Names: '{name}'" if name != "Unknown" else "Names: Unknown",
                f"Center Position: Unknown"
            ]
            
            control_info = "- " + "; ".join(info_parts) + "."
            output.append(control_info)
        
        output.append("")  # For an empty line between categories

    return "\n".join(output).strip()


def analyze_app(application_name_contains=None, size_category=None, additional_search_options=None):
    root = auto.GetRootControl()

    control = None
    if application_name_contains:
        for win in root.GetChildren():
            if application_name_contains.lower() in win.Name.lower():
                control = win
                break
        if not control:
            return f'Window containing "{application_name_contains}" not found.'
    else:
        control = root

    if not control.Exists(0, 0):
        return f'Application with title containing "{application_name_contains}" is not running or window not found.'

    search_strings = additional_search_options.lower().split(',') if additional_search_options else []
    search_strings = [s.strip() for s in search_strings if s.strip()]

    matched_controls_with_rects, unmatched_controls_with_rects = walk_control(control, control_type=None, search_strings=search_strings)

    output = ""
    if matched_controls_with_rects:
        output += f"Matched Elements:\n{sort_and_categorize_rects(matched_controls_with_rects, size_category_to_print=size_category)}\n"
    if unmatched_controls_with_rects:
        output += f"All Elements:\n{sort_and_categorize_rects(unmatched_controls_with_rects, size_category_to_print=size_category)}"
    return output


# Example Usage
if __name__ == '__main__':
    search_options = "contenteditable"
    search_terms = search_options.replace('', '').strip()
    print(search_terms)
    print(analyze_app(application_name_contains='Firefox', additional_search_options=search_terms))
