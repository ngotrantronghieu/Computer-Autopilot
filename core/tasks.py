import json

TASKS_FILE = "tasks.json"

def load_tasks():
    """Load tasks from JSON file"""
    try:
        with open(TASKS_FILE, "r") as f:
            tasks = json.load(f)
            # Convert legacy format to new format
            for name, data in tasks.items():
                if not isinstance(data, dict):
                    tasks[name] = {
                        'description': '',
                        'actions': data
                    }
            return tasks
    except FileNotFoundError:
        return {}

def save_tasks(tasks):
    """Save tasks to JSON file"""
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=2)

def save_task(name, task_data):
    """
    Save a single task
    task_data can be either:
    - A dict with 'description' and 'actions' keys
    - A list of actions (legacy format)
    """
    tasks = load_tasks()
    if isinstance(task_data, list):
        # Convert legacy format
        task_data = {
            'description': '',
            'actions': task_data
        }
    tasks[name] = task_data
    save_tasks(tasks)

def delete_task(name):
    """Delete a task"""
    tasks = load_tasks()
    if name in tasks:
        del tasks[name]
        save_tasks(tasks)
        return True
    return False

def get_task(name):
    """
    Get task by name
    Returns either:
    - A dict with 'description' and 'actions' keys
    - A list of actions (legacy format)
    """
    tasks = load_tasks()
    task_data = tasks.get(name, {'description': '', 'actions': []})
    return task_data
