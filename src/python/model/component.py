
# First tier component classes
class Component:
    def __init__(self):
        pass

# Second tier component classes
class DiffuseComponent(Component):
    pass

class PointSourceComponent(Component):
    pass

class TemplateComponent(Component):
    pass

# Third tier component classes
class CMB(DiffuseComponent):
    pass

class RadioSource(PointSourceComponent):
    pass

class CMBRelQuad(TemplateComponent):
    pass