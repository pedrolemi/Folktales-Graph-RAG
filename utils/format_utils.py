from models.agent import Agent
from models.object import Object
from models.place import Place

def _format_hierarchy(hierarchy: dict, n_tabs: int):
	lines = []
	for name, info in hierarchy.items():
		line = "\t"*n_tabs+f"- {name}"

		description = info.get("description")
		if description:
			line += f": {description}"

		lines.append(line)

		children = info.get("children")
		if children:
			lines.append(_format_hierarchy(children, n_tabs + 1))
	return "\n".join(lines)

def format_hierarchy(hierarchy: dict):
	root = next(iter(hierarchy.values()))
	children = root.get("children", {})
	return _format_hierarchy(children, 0)

def _format_classes(hierarchy: dict):
	names = []
	for name, info in hierarchy.items():
		names.append(name)

		children = info.get("children")
		if children:
			names.extend(_format_classes(children))
	return names

def format_classes(hierarchy: dict):
	root = next(iter(hierarchy.values()))
	children = root.get("children", {})
	return ", ".join(name for name in _format_classes(children))

def format_agents(agents: list[Agent]):
	formatted_agents = []
	for i, agent in enumerate(agents):
		instance_name = agent.instance_name

		if agent.name:
			formatted_agents.append(f"- {i}. {instance_name} ({agent.name})")
		else:
			formatted_agents.append(f"- {i}. {instance_name}")
	return "\n".join(formatted_agents)

def format_places(places: list[Place]):
	return "\n".join(f"- {i}. {place.instance_name}" for i, place in enumerate(places))

def format_objects(objects: list[Object]):
	return "\n".join(f"- {i}. {object.instance_name}" for i, object in enumerate(objects))