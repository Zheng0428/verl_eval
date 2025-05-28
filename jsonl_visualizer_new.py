import gradio as gr
import json
import os
from typing import List, Dict, Any, Tuple, Optional

def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return its contents as a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                data.append({"error": f"Could not parse line: {line[:100]}..."})
    return data

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSON file and return its contents as a list of dictionaries."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # If the JSON is an array/list, return it directly
        if isinstance(data, list):
            return data
        # If it's a single object, wrap it in a list
        elif isinstance(data, dict):
            return [data]
        # If it's another type, wrap it in a dict and then in a list
        else:
            return [{"value": data}]
    except json.JSONDecodeError as e:
        return [{"error": f"Failed to parse JSON: {str(e)}"}]

def load_file_content(file_path: str) -> Tuple[List[Dict[str, Any]], str]:
    """Detect file type and load content accordingly."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.jsonl':
        return load_jsonl_file(file_path), 'jsonl'
    elif file_extension == '.json':
        return load_json_file(file_path), 'json'
    else:
        # Try to detect the format by looking at the content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                # If the first line is a valid JSON and there are more lines, assume JSONL
                try:
                    json.loads(first_line)
                    second_line = f.readline().strip()
                    if second_line:
                        return load_jsonl_file(file_path), 'jsonl'
                    else:
                        # Only one line, could be a JSON array/object on a single line
                        return load_json_file(file_path), 'json'
                except json.JSONDecodeError:
                    # If the first line isn't valid JSON, try loading as complete JSON
                    return load_json_file(file_path), 'json'
        except Exception:
            # If all detection fails, try both formats
            try:
                return load_json_file(file_path), 'json'
            except:
                return load_jsonl_file(file_path), 'jsonl'

def get_file_info(file_path: str) -> Tuple[int, List[str], str]:
    """Get the number of lines and a preview of keys in the file."""
    if not os.path.exists(file_path):
        return 0, ["File not found"], ""
    
    data, file_type = load_file_content(file_path)
    keys = []
    if data and isinstance(data[0], dict):
        keys = list(data[0].keys())
    
    return len(data), keys, file_type

def display_line(file_path: str, line_index: int) -> Tuple[str, str, str]:
    """Display a specific item from the JSON/JSONL file."""
    if not os.path.exists(file_path):
        return "File not found", "", ""
    
    data, file_type = load_file_content(file_path)
    
    if line_index < 0 or line_index >= len(data):
        return f"Item index out of range (0-{len(data)-1})", "", ""
    
    current_item = data[line_index]
    
    # Pretty print the raw JSON
    raw_json = json.dumps(current_item, indent=2)
    
    # Format all keys consistently
    formatted_content = ""
    for key, value in current_item.items():
        formatted_content += f"## {key}:\n"
        if isinstance(value, str):
            formatted_content += value + "\n\n"
        else:
            formatted_content += json.dumps(value, indent=2) + "\n\n"
    
    item_type = "Line" if file_type == "jsonl" else "Item"
    return f"{item_type} {line_index+1} of {len(data)}", raw_json, formatted_content

def apply_filters(data: List[Dict[str, Any]], filters: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Apply filters to the data and return filtered results."""
    if not filters:
        return data
    
    filtered_data = []
    for item in data:
        matches_all = True
        for filter_rule in filters:
            key = filter_rule.get('key', '')
            operator = filter_rule.get('operator', '')
            value = filter_rule.get('value', '')
            
            if not key or not operator or key not in item:
                continue
                
            item_value = str(item[key])
            if operator == 'contains' and value.lower() not in item_value.lower():
                matches_all = False
            elif operator == 'equals' and value.lower() != item_value.lower():
                matches_all = False
            elif operator == 'not_contains' and value.lower() in item_value.lower():
                matches_all = False
            elif operator == 'not_equals' and value.lower() == item_value.lower():
                matches_all = False
        
        if matches_all:
            filtered_data.append(item)
    
    return filtered_data

def create_interface():
    """Create the Gradio interface for the JSON/JSONL visualizer."""
    with gr.Blocks(title="JSON/JSONL Visualizer") as app:
        # Store filters in state
        filter_state = gr.State([])
        file_type_state = gr.State("")
        
        gr.Markdown("# JSON/JSONL Visualizer")
        gr.Markdown("Upload a JSON or JSONL file or provide a path to view its contents.")
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload JSON/JSONL File")
                path_input = gr.Textbox(label="or Enter File Path")
                load_btn = gr.Button("Load File")
                
                info_text = gr.Textbox(label="File Info", interactive=False)
                
                # Add filtering section
                gr.Markdown("### Filtering Rules")
                with gr.Row():
                    filter_key = gr.Textbox(label="Key", scale=1)
                    filter_operator = gr.Dropdown(
                        choices=["contains", "equals", "not_contains", "not_equals"],
                        label="Operator",
                        value="contains",
                        scale=1
                    )
                    filter_value = gr.Textbox(label="Value", scale=1)
                
                with gr.Row():
                    add_filter_btn = gr.Button("Add Filter")
                    clear_filters_btn = gr.Button("Clear Filters")
                
                filter_display = gr.Markdown(label="Active Filters")
                
                with gr.Row():
                    prev_btn = gr.Button("← Previous")
                    line_input = gr.Number(label="Line Number", value=1, precision=0)
                    next_btn = gr.Button("Next →")
            
            with gr.Column(scale=2):
                line_info = gr.Textbox(label="Item Information", interactive=False)
                
                with gr.Tabs():
                    with gr.TabItem("Formatted View"):
                        formatted_output = gr.Markdown()
                    with gr.TabItem("Raw JSON"):
                        raw_output = gr.Code(language="json")
        
        # Update the load_file function to handle both formats and filters
        def load_file(file_obj, file_path, filters):
            if file_obj is not None and hasattr(file_obj, "name"):
                path = file_obj.name
            elif file_path:
                path = file_path
            else:
                return "No file selected", 1, "No file loaded", "", "", filters, ""
            
            data, file_type = load_file_content(path)
            filtered_data = apply_filters(data, filters)
            num_items = len(filtered_data)
            keys = list(filtered_data[0].keys()) if filtered_data else []
            
            item_type = "lines" if file_type == "jsonl" else "items"
            info_msg = f"{num_items} {item_type} (filtered from {len(data)} total), Keys: {', '.join(keys)}"
            
            return (
                info_msg, 
                1, 
                *display_line_filtered(path, 0, filters),
                filters,
                file_type
            )
        
        def display_line_filtered(file_path: str, line_index: int, filters: List[Dict[str, str]]) -> Tuple[str, str, str]:
            """Display a specific item from the filtered JSON/JSONL data."""
            if not os.path.exists(file_path):
                return "File not found", "", ""
            
            data, file_type = load_file_content(file_path)
            filtered_data = apply_filters(data, filters)
            
            if not filtered_data:
                return "No matching records found", "", ""
            
            if line_index < 0 or line_index >= len(filtered_data):
                return f"Item index out of range (0-{len(filtered_data)-1})", "", ""
            
            current_item = filtered_data[line_index]
            raw_json = json.dumps(current_item, indent=2)
            
            formatted_content = ""
            for key, value in current_item.items():
                formatted_content += f"## {key}:\n"
                if isinstance(value, str):
                    formatted_content += value + "\n\n"
                else:
                    formatted_content += json.dumps(value, indent=2) + "\n\n"
            
            item_type = "Line" if file_type == "jsonl" else "Item"
            return f"{item_type} {line_index+1} of {len(filtered_data)}", raw_json, formatted_content
        
        # Add filter management functions
        def add_filter(key, operator, value, current_filters):
            if key and operator and value:
                new_filters = current_filters + [{"key": key, "operator": operator, "value": value}]
                filter_text = format_filter_display(new_filters)
                return new_filters, filter_text, "", "contains", ""
            return current_filters, format_filter_display(current_filters), key, operator, value

        def clear_filters(current_filters):
            return [], "", "", "contains", ""

        def format_filter_display(filters):
            if not filters:
                return "No active filters"
            return "\n".join([
                f"- {f['key']} {f['operator']} '{f['value']}'"
                for f in filters
            ])

        # Update event handlers
        load_btn.click(
            load_file,
            inputs=[file_input, path_input, filter_state],
            outputs=[info_text, line_input, line_info, raw_output, formatted_output, filter_state, file_type_state]
        )

        add_filter_btn.click(
            add_filter,
            inputs=[filter_key, filter_operator, filter_value, filter_state],
            outputs=[filter_state, filter_display, filter_key, filter_operator, filter_value]
        )

        clear_filters_btn.click(
            clear_filters,
            inputs=[filter_state],
            outputs=[filter_state, filter_display, filter_key, filter_operator, filter_value]
        )

        # Update navigation to handle filters and both file types
        def navigate_filtered(file_obj, file_path, line_num, direction, filters):
            if file_obj is not None and hasattr(file_obj, "name"):
                path = file_obj.name
            elif file_path:
                path = file_path
            else:
                return 1, "No file loaded", "", ""
            
            data, _ = load_file_content(path)
            filtered_data = apply_filters(data, filters)
            new_line = max(1, min(len(filtered_data), line_num + direction))
            
            return new_line, *display_line_filtered(path, new_line - 1, filters)

        prev_btn.click(
            lambda file, path, line, filters: navigate_filtered(file, path, line, -1, filters),
            inputs=[file_input, path_input, line_input, filter_state],
            outputs=[line_input, line_info, raw_output, formatted_output]
        )

        next_btn.click(
            lambda file, path, line, filters: navigate_filtered(file, path, line, 1, filters),
            inputs=[file_input, path_input, line_input, filter_state],
            outputs=[line_input, line_info, raw_output, formatted_output]
        )

        # Update direct line access to handle filters and both file types
        line_input.change(
            lambda file, path, line, filters: display_line_filtered(
                file.name if file and hasattr(file, "name") else path,
                int(line) - 1,
                filters
            ),
            inputs=[file_input, path_input, line_input, filter_state],
            outputs=[line_info, raw_output, formatted_output]
        )

    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch()
