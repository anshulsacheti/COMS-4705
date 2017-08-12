class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        # raise NotImplementedError('Please implement left_arc!')
        # return -1

        # left_arcL
        # Add the arc (b, L, s) to A, and pop Σ. That is, draw an arc between
        # the next node on the buffer and the next node on the stack, with the label L.

        # A LEFT-ARC transition (for any dependency label l) adds the arc (b,l,s) to A,
        # where s is the node on top of the stack and b is the first node in the buffer,
        # and pops the stack. It has as a precondition that the token s is not the
        # artificial root node 0 and does not already have a head.

        # A RIGHT-ARCl transition (for any dependency label l) adds the arc (s, l, b)
        # to A, where s is the node on top of the stack and b is the first node in the
        # buffer, and pushes the node b onto the stack.

        if not conf.buffer or not conf.stack:
            return -1

        s = conf.stack[-1]

        #Check if root node
        if s==0:
            return -1

        #Check if node already has head
        for arc in conf.arcs:
            if arc[2]==s:
                return -1

        #Pop node off top of stack
        idx_wi = conf.stack.pop(-1)

        #First node in buffer
        idx_wj = conf.buffer[0]

        #Add Arc
        conf.arcs.append((idx_wj,relation,idx_wi))


    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """

        # right_arcL
        # Add the arc (s, L, b) to A, and push b onto Σ.

        if not conf.buffer or not conf.stack:
            return -1

        # You get this one for free! Use it as an example.

        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer.pop(0)

        conf.stack.append(idx_wj)
        conf.arcs.append((idx_wi, relation, idx_wj))

    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """

        # reduce
        # pop Σ

        # The REDUCE transition pops the stack and is subject to the preconditions
        # that the top token has a head.

        # raise NotImplementedError('Please implement reduce!')
        # return -1

        if not conf.stack:
            return -1

        s = conf.stack[-1]

        #Check if node already has head
        headFound = False
        for arc in conf.arcs:
            if arc[2]==s:
                conf.stack.pop(-1)
                headFound = True
                break

        if not headFound:
            return -1

    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """

        # shift
        # Remove b from B and add it to Σ.

        # The SHIFT transition removes the first node in the buffer and pushes it
        # onto the stack.

        if not conf.buffer:
            return -1

        b = conf.buffer.pop(0)
        conf.stack.append(b)
        # raise NotImplementedError('Please implement shift!')
        # return -1
