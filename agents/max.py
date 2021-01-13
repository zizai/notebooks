import h5py
import numpy as np
from matplotlib import pyplot as plt


class Generator:
    """
    Primary function of this generator is to generate variable length
    dataset for training variable length programs. It creates a generator
    object for every length of program that you want to generate. This
    process allow finer control of every batch that you feed into the
    network. This class can also be used in fixed length training.
    """
    def __init__(self,
                 data_labels_paths,
                 batch_size=32,
                 time_steps=3,
                 stack_size=2,
                 canvas_shape=[64, 64, 64],
                 primitives=None):
        """
        :param stack_size: maximum size of stack used for programs.
        :param canvas_shape: canvas shape
        :param primitives: Dictionary containing pre-rendered shape primitives in the
         form of grids.
        :param time_steps: Max time steps for generated programs
        :param data_labels_paths: dictionary containing paths for different
        lengths programs
        :param batch_size: batch_size
        """
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.canvas_shape = canvas_shape
        self.stack_size = stack_size

        self.programs = {}
        self.data_labels_path = data_labels_paths
        for index in data_labels_paths.keys():
            with open(data_labels_paths[index]) as data_file:
                self.programs[index] = data_file.readlines()
        all_programs = []

        for k in self.programs.keys():
            all_programs += self.programs[k]

        self.unique_draw = self.get_draw_set(all_programs)
        self.unique_draw.sort()
        # Append ops in the end and the last one is for stop symbol
        self.unique_draw += ["+", "*", "-", "$"]
        sim = SimulateStack(self.time_steps // 2 + 1, self.canvas_shape,
                            self.unique_draw)

        if not (type(primitives) is dict):
            # # Draw all primitive in one go and reuse them later
            # sim.draw_all_primitives(self.unique_draw)
            # self.primitives = sim.draw_all_primitives(self.unique_draw)
            # dd.io.save("mix_len_all_primitives.h5", self.primitives)
            self.primitives = h5py.File('data/primitives.h5', 'r')
        else:
            self.primitives = primitives
        self.parser = Parser()

    def parse(self, expression):
        """
        NOTE: This method is different from parse method in Parser class
        Takes an expression, returns a serial program
        :param expression: program expression in postfix notation
        :return program:
        """
        self.shape_types = ["u", "p", "y"]
        self.op = ["*", "+", "-"]
        program = []
        for index, value in enumerate(expression):
            if value in self.shape_types:
                program.append({})
                program[-1]["type"] = "draw"

                # find where the parenthesis closes
                close_paren = expression[index:].index(")") + index
                program[-1]["value"] = expression[index:close_paren + 1]
            elif value in self.op:
                program.append({})
                program[-1]["type"] = "op"
                program[-1]["value"] = value
            else:
                pass
        return program

    def get_draw_set(self, expressions):
        """
        Find a sorted set of draw type from the entire dataset. The idea is to
        use only the plausible position, scale and shape combinations and
        reject that are not possible because of the restrictions we have in
        the dataset.
        :param expressions: List containing entire dataset in the form of
        expressions.
        :return: unique_chunks: Unique sorted draw operations in the dataset.
        """
        shapes = ["u", "p", "y"]
        chunks = []
        for expression in expressions:
            for i, e in enumerate(expression):
                if e in shapes:
                    index = i
                    last_index = expression[index:].index(")")
                    chunks.append(expression[index:index + last_index + 1])
        return list(set(chunks))

    def get_train_data(self, batch_size: int, program_len: int,
                       final_canvas=False, if_randomize=True, if_primitives=False,
                       num_train_images=400, if_jitter=False):
        """
        This is a special generator that can generate dataset for any length.
        This essentially corresponds to the "variable len program"
        experiment. Here, generate a dataset for training for fixed length.
        Since, this is a generator, you need to make a generator object for
        all different kind of lengths and use them as required. It is made
        sure that samples are shuffled only once in an epoch and all the
        samples are different in an epoch.
        :param if_randomize: whether to randomize the training instance during training.
        :param if_primitives: if pre-rendered primitives are given
        :param num_train_images: Number of training instances
        :param if_jitter: whether to jitter the voxels or not
        :param batch_size: batch size for the current program
        :param program_len: which program length dataset to sample
        :param final_canvas: This is special mode of data generation where
        all the dataset is loaded in one go and iteratively yielded. The
        dataset for only target images is created.
        """
        # The last label corresponds to the stop symbol and the first one to
        # start symbol.
        labels = np.zeros((batch_size, program_len + 1), dtype=np.int64)
        sim = SimulateStack(program_len // 2 + 1, self.canvas_shape,
                            self.unique_draw)
        sim.get_all_primitives(self.primitives)
        parser = Parser()

        if final_canvas:
            # We will load all the final canvases from the disk.
            path = self.data_labels_path[program_len]
            path = path[0:-15]
            Stack = np.zeros((1, num_train_images, 1, self.canvas_shape[0],
                              self.canvas_shape[1], self.canvas_shape[2]),
                             dtype=np.bool)
            for i in range(num_train_images):
                p = path + "{}.png".format(i + 1)
                img = plt.imread(p)[:, :, 0]
                Stack[0, i, 0, :, :] = img.astype(np.bool)

        while True:
            # Random things to select random indices
            IDS = np.arange(num_train_images)
            if if_randomize:
                np.random.shuffle(IDS)
            for rand_id in range(0, num_train_images - batch_size,
                                 batch_size):
                image_ids = IDS[rand_id:rand_id + batch_size]
                if not final_canvas:
                    stacks = []
                    sampled_exps = []
                    for index, value in enumerate(image_ids):
                        sampled_exps.append(self.programs[program_len][value])
                        if not if_primitives:
                            program = parser.parse(
                                self.programs[program_len][value])
                        else:
                            # if all primitives are give already, parse using
                            #  different parser to get the keys to dict
                            program = self.parse(self.programs[program_len][
                                                         value])
                        sim.generate_stack(program, if_primitives=if_primitives)
                        stack = sim.stack_t
                        stack = np.stack(stack, axis=0)
                        stacks.append(stack)
                    stacks = np.stack(stacks, 1).astype(dtype=np.float32)
                else:
                    # When only target image is required
                    stacks = Stack[0:1, image_ids, 0:1, :, :, :].astype(
                        dtype=np.float32)
                for index, value in enumerate(image_ids):
                    # Get the current program
                    exp = self.programs[program_len][value]
                    program = self.parse(exp)
                    for j in range(program_len):
                        labels[index, j] = self.unique_draw.index(
                            program[j]["value"])

                    labels[:, -1] = len(self.unique_draw) - 1

                if if_jitter:
                    temp = stacks[-1, :, 0, :, :, :]
                    stacks[-1, :, 0, :, :, :] = np.roll(temp, (np.random.randint(-3, 4),
                                                               np.random.randint(-3, 4),
                                                               np.random.randint(-3, 4)),
                                                        axis=(1, 2, 3))

                yield [stacks, labels]

    def get_test_data(self, batch_size: int, program_len: int,
                      if_randomize=False, final_canvas=False,
                      num_train_images=None, num_test_images=None,
                      if_primitives=False, if_jitter=False):
        """
        Test dataset creation. It is assumed that the first num_training
        examples in the dataset corresponds to training and later num_test
        are validation dataset. The validation may optionally be shuffled
        randomly but usually not required.
        :param num_train_images:
        :param if_primitives: if pre-rendered primitives are given
        :param if_jitter: Whether to jitter the voxel grids
        :param num_test_images: Number of test images
        :param batch_size: batch size of dataset to yielded
        :param program_len: length of program to be generated
        :param if_randomize: if randomize
        :param final_canvas: if true return only the target canvas instead of
        complete stack to save memory
        :return:
        """
        # This generates test data of fixed length. Samples are not shuffled
        # by default.
        labels = np.zeros((batch_size, program_len + 1), dtype=np.int64)
        sim = SimulateStack(program_len // 2 + 1, self.canvas_shape,
                            self.unique_draw)
        sim.get_all_primitives(self.primitives)
        parser = Parser()

        if final_canvas:
            # We will load all the final canvases from the disk.
            path = self.data_labels_path[program_len]
            path = path[0:-15]
            Stack = np.zeros((1, num_test_images, 1, self.canvas_shape[0],
                              self.canvas_shape[1], self.canvas_shape[2]),
                             dtype=np.bool)
            for i in range(num_train_images,
                           num_test_images + num_train_images):
                p = path + "{}.png".format(i + 1)
                img = plt.imread(p)[:, :, 0]
                Stack[0, i, 0, :, :] = img.astype(np.bool)

        while True:
            # Random things to select random indices
            IDS = np.arange(num_train_images, num_train_images +
                            num_test_images)
            if if_randomize:
                np.random.shuffle(IDS)
            for rand_id in range(0, num_test_images - batch_size, batch_size):
                image_ids = IDS[rand_id: rand_id + batch_size]
                if not final_canvas:
                    stacks = []
                    sampled_exps = []
                    for index, value in enumerate(image_ids):
                        sampled_exps.append(self.programs[program_len][value])
                        if not if_primitives:
                            program = parser.parse(
                                self.programs[program_len][value])
                        if True:
                            # if all primitives are give already, parse using
                            #  different parser to get the keys to dict
                            try:
                                program = self.parse(self.programs[program_len][
                                                         value])
                            except:
                                print(index, self.programs[program_len][
                                    value])
                        sim.generate_stack(program, if_primitives=if_primitives)
                        stack = sim.stack_t
                        stack = np.stack(stack, axis=0)
                        stacks.append(stack)
                    stacks = np.stack(stacks, 1).astype(dtype=np.float32)
                else:
                    # When only target image is required
                    stacks = Stack[0:1, image_ids, 0:1, :, :, :].astype(
                        dtype=np.float32)
                for index, value in enumerate(image_ids):
                    # Get the current program
                    exp = self.programs[program_len][value]
                    program = self.parse(exp)
                    for j in range(program_len):
                        try:
                            labels[index, j] = self.unique_draw.index(
                                program[j]["value"])
                        except:
                            print(program)

                    labels[:, -1] = len(self.unique_draw) - 1

                if if_jitter:
                    temp = stacks[-1, :, 0, :, :, :]
                    stacks[-1, :, 0, :, :, :] = np.roll(temp, (np.random.randint(-3, 4),
                                                               np.random.randint(-3, 4),
                                                               np.random.randint(-3, 4)),
                                                        axis=(1, 2, 3))
                yield [stacks, labels]


class Parser:
    """
    Parser to parse the program written in postfix notation
    """

    def __init__(self):
        self.shape_types = ["u", "p", "y"]
        self.op = ["*", "+", "-"]

    def parse(self, expression: str):
        """
        Takes an empression, returns a serial program
        :param expression: program expression in postfix notation
        :return program:
        """
        program = []
        for index, value in enumerate(expression):
            if value in self.shape_types:
                # draw shape instruction
                program.append({})
                program[-1]["value"] = value
                program[-1]["type"] = "draw"
                # find where the parenthesis closes
                close_paren = expression[index:].index(")") + index

                program[-1]["param"] = expression[index + 2:close_paren].split(
                    ",")
                if program[-1]["param"][0][0] == "(":
                    print (expression)

            elif value in self.op:
                # operations instruction
                program.append({})
                program[-1]["type"] = "op"
                program[-1]["value"] = value

            elif value == "$":
                # must be a stop symbol
                program.append({})
                program[-1]["type"] = "stop"
                program[-1]["value"] = "$"
        return program


class PushDownStack(object):
    """Simple PushDown Stack implements in the form of array"""

    def __init__(self, max_len, canvas_shape):
        _shape = [max_len] + canvas_shape
        self.max_len = max_len
        self.canvas_shape = canvas_shape
        self.items = []
        self.max_len = max_len

    def push(self, item):
        if len(self.items) >= self.max_len:
            assert False, "exceeds max len for stack!!"
        self.items = [item.copy()] + self.items

    def pop(self):
        if len(self.items) == 0:
            assert False, "below min len of stack!!"
        item = self.items[0]
        self.items = self.items[1:]
        return item

    def get_items(self):
        """
        In this we create a fixed shape tensor amenable for further usage
        :return:
        """
        size = [self.max_len] + self.canvas_shape
        stack_elements = np.zeros(size, dtype=bool)
        length = len(self.items)
        for j in range(length):
            stack_elements[j, :, :, :] = self.items[j]
        return stack_elements

    def clear(self):
        """Re-initializes the stack"""
        self.items = []


class SimulateStack:
    """
    Simulates the stack for CSG
    """
    def __init__(self, max_len, canvas_shape, draw_uniques):
        """
        :param max_len: max size of stack
        :param canvas_shape: canvas shape
        :param draw_uniques: unique operations (draw + ops)
        """
        self.draw_obj = Draw(canvas_shape=canvas_shape)
        self.draw = {
            "u": self.draw_obj.draw_cube,
            "p": self.draw_obj.draw_sphere,
            "y": self.draw_obj.draw_cylinder
        }
        self.op = {"*": self._and, "+": self._union, "-": self._diff}
        self.stack = PushDownStack(max_len, canvas_shape)
        self.stack_t = []
        self.stack.clear()
        self.stack_t.append(self.stack.get_items())
        self.parser = Parser()

    def draw_all_primitives(self, draw_uniques):
        """
        Draws all primitives so that we don't have to draw them over and over.
        :param draw_uniques: unique operations (draw + ops)
        :return:
        """
        self.primitives = {}
        for index, value in enumerate(draw_uniques[0:-4]):
            p = self.parser.parse(value)[0]
            which_draw = p["value"]
            if which_draw == "u" or which_draw == "p":
                # draw cube or sphere
                x = int(p["param"][0])
                y = int(p["param"][1])
                z = int(p["param"][2])
                radius = int(p["param"][3])
                layer = self.draw[which_draw]([x, y, z], radius)

            elif which_draw == "y":
                # draw cylinder
                # TODO check if the order is correct.
                x = int(p["param"][0])
                y = int(p["param"][1])
                z = int(p["param"][2])
                radius = int(p["param"][3])
                height = int(p["param"][4])
                layer = self.draw[p["value"]]([x, y, z], radius, height)
            self.primitives[value] = layer
        return self.primitives

    def get_all_primitives(self, primitives):
        """ Get all primitive from outseide class
        :param primitives: dictionary containing pre-rendered shape primitives
        """
        self.primitives = primitives

    def parse(self, expression):
        """
        NOTE: This method generates terminal symbol for an input program expressions.
        :param expression: program expression in postfix notation
        :return program:
        """
        shape_types = ["u", "p", "y"]
        op = ["*", "+", "-"]
        program = []
        for index, value in enumerate(expression):
            if value in shape_types:
                program.append({})
                program[-1]["type"] = "draw"

                # find where the parenthesis closes
                close_paren = expression[index:].index(")") + index
                program[-1]["value"] = expression[index:close_paren + 1]
            elif value in op:
                program.append({})
                program[-1]["type"] = "op"
                program[-1]["value"] = value
            else:
                pass
        return program

    def generate_stack(self, program: list, start_scratch=True, if_primitives=False):
        """
        Executes the program step-by-step and stores all intermediate stack
        states.
        :param if_primitives: if pre-rendered primitives are given.
        :param program: List with each item a program step
        :param start_scratch: whether to start creating stack from scratch or
        stack already exist and we are appending new instructions. With this
        set to False, stack can be started from its previous state.
        """
        # clear old garbage
        if start_scratch:
            self.stack_t = []
            self.stack.clear()
            self.stack_t.append(self.stack.get_items())

        for index, p in enumerate(program):
            if p["type"] == "draw":
                if if_primitives:
                    # fast retrieval of shape primitive
                    layer = self.primitives[p["value"]]
                    self.stack.push(layer)
                    self.stack_t.append(self.stack.get_items())
                    continue

                if p["value"] == "u" or p["value"] == "p":
                    # draw cube or sphere
                    x = int(p["param"][0])
                    y = int(p["param"][1])
                    z = int(p["param"][2])
                    radius = int(p["param"][3])
                    layer = self.draw[p["value"]]([x, y, z], radius)

                elif p["value"] == "y":
                    # draw cylinder
                    # TODO check if the order is correct.
                    x = int(p["param"][0])
                    y = int(p["param"][1])
                    z = int(p["param"][2])
                    radius = int(p["param"][3])
                    height = int(p["param"][4])
                    layer = self.draw[p["value"]]([x, y, z], radius, height)
                self.stack.push(layer)

                # Copy to avoid orver-write
                # self.stack_t.append(self.stack.items.copy())
                self.stack_t.append(self.stack.get_items())
            else:
                # operate
                obj_2 = self.stack.pop()
                obj_1 = self.stack.pop()
                layer = self.op[p["value"]](obj_1, obj_2)
                self.stack.push(layer)
                # Copy to avoid over-write
                # self.stack_t.append(self.stack.items.copy())
                self.stack_t.append(self.stack.get_items())

    def _union(self, obj1, obj2):
        """Union between voxel grids"""
        return np.logical_or(obj1, obj2)

    def _and(self, obj1, obj2):
        """Intersection between voxel grids"""
        return np.logical_and(obj1, obj2)

    def _diff(self, obj1, obj2):
        """Subtraction between voxel grids"""
        return (obj1 * 1. - np.logical_and(obj1, obj2) * 1.).astype(np.bool)


class Draw:
    def __init__(self, canvas_shape=[64, 64, 64]):
        """
        Helper Class for drawing the canvases.
        :param canvas_shape: shape of the canvas on which to draw objects
        """
        self.canvas_shape = canvas_shape

    def draw_sphere(self, center, radius):
        """Makes sphere inside a cube of canvas_shape
        :param center: center of the sphere
        :param radius: radius of sphere
        :return:
        """
        radius -= 1
        canvas = np.zeros(self.canvas_shape, dtype=bool)
        for x in range(center[0] - radius, center[0] + radius + 1):
            for y in range(center[1] - radius, center[1] + radius + 1):
                for z in range(center[2] - radius, center[2] + radius + 1):
                    if np.linalg.norm(np.array(center) - np.array(
                            [x, y, z])) <= radius:
                        canvas[x, y, z] = True
        return canvas

    def draw_cube(self, center, side):
        """Makes cube inside a cube of canvas_shape
        :param center: center of cube
        :param side: side of cube
        :return:
        """
        side -= 1
        canvas = np.zeros(self.canvas_shape, dtype=bool)
        side = side // 2
        for x in range(center[0] - side, center[0] + side + 1):
            for y in range(center[1] - side, center[1] + side + 1):
                for z in range(center[2] - side, center[2] + side + 1):
                    canvas[x, y, z] = True
        return canvas

    def draw_cylinder(self, center, radius, height):
        """Makes cylinder inside a of canvas_shape
        :param center: center of cylinder
        :param radius: radius of cylinder
        :param height: height of cylinder
        :return:
        """
        radius -= 1
        height -= 1
        canvas = np.zeros(self.canvas_shape, dtype=bool)

        for z in range(center[2] - int(height / 2),
                       center[2] + int(height / 2) + 1):
            for x in range(center[0] - radius, center[0] + radius + 1):
                for y in range(center[1] - radius, center[1] + radius + 1):
                    if np.linalg.norm(
                                    np.array([center[0], center[1]]) - np.array(
                                [x, y])) <= radius:
                        canvas[x, y, z] = True
        return canvas
