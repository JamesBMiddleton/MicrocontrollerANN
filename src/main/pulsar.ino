Pulsar::Pulsar()
    :_is_f_pulsing{false}, _is_b_pulsing{false}, _pulse_timer{0}, 
    _pulse_length{120}, _pulse_step{1},
    _relay_threshold{(uint8_t)(_pulse_length/2)}, _max_brightness{MAX_BRIGHTNESS},
    _brightness{MIN_BRIGHTNESS},
    _bright_step{(_max_brightness - _brightness) / _pulse_length}, _hue{0}, _sat{0} 
{}

void Pulsar::update()
{
    _pulse_timer += _pulse_step;
    _brightness += _bright_step; // sacrificing memory for performance.
    if (_pulse_timer == 0) 
    {
        _is_f_pulsing = false;
        _is_b_pulsing = false;
        _pulse_step = -_pulse_step;
        _bright_step = -_bright_step;
    }
    if (_pulse_timer == _pulse_length)
    {
        _pulse_step = -_pulse_step;
        _bright_step = -_bright_step;
    }
}

void Pulsar::set_max_brightness(const uint8_t& new_max)
{
#ifdef DEBUG
    if (_is_f_pulsing || _is_b_pulsing)
        error = "max brightness set during pulse.";
    if (new_max < MIN_BRIGHTNESS)
        error = "max brightness set below MIN_BRIGHTNESS.";
    if (new_max > MAX_BRIGHTNESS)
        error = "max brightness set above MAX_BRIGHTNESS.";
#endif
    _max_brightness = new_max;
    _bright_step = (_max_brightness - _brightness) / _pulse_length;
}


NodePulsar::NodePulsar(const uint8_t& x, const uint8_t& y, 
                        const uint8_t& radius) 
    :Pulsar{}, _x{x}, _y{y}, _radius{radius}
{}

void NodePulsar::draw()
{
    matrix.fillCircle(_x, _y, _radius, matrix.colorHSV(_hue, _sat, _brightness));
}

void NodePulsar::update()
{
    if (_is_f_pulsing)
    {
        Pulsar::update();
        if (_pulse_timer == _relay_threshold && _pulse_step > 0)
            for (uint8_t i{0}; i<_f_links.size(); ++i)
                _f_links[i]->init_f_pulse();
    }
    if (_is_b_pulsing)
    {
        Pulsar::update();
        if (_pulse_timer == _relay_threshold && _pulse_step > 0)
            for (uint8_t i{0}; i<_b_links.size(); ++i)
                _b_links[i]->init_b_pulse();
    }
}

LinkPulsar::LinkPulsar(const uint8_t& x1, const uint8_t& y1,
                        const uint8_t& x2, const uint8_t& y2, 
                        NodePulsar* forward_node, NodePulsar* backward_node)
    :Pulsar{}, _x1{x1}, _y1{y1}, _x2{x2}, _y2{y2}, _forward_node{forward_node},
    _backward_node{backward_node}
{}

void LinkPulsar::update()
{
    if (_is_f_pulsing)
    {
        Pulsar::update();
        if (_pulse_timer == _relay_threshold && _pulse_step > 0)
            _forward_node->init_f_pulse();
    }
    if (_is_b_pulsing)
    {
        Pulsar::update();
        if (_pulse_timer == _relay_threshold && _pulse_step > 0)
            _backward_node->init_b_pulse();
    }
}

void LinkPulsar::draw()
{
    matrix.drawLine(_x1, _y1, _x2, _y2, matrix.colorHSV(_hue, _sat, _brightness));
}



void link_nodes(NodePulsar* node1, NodePulsar* node2, LinkPulsar* link)
{
    *link = LinkPulsar{node1->get_x(), node1->get_y(), node2->get_x(), 
                        node2->get_y(), node2, node1};
    node1-> add_forwardlink(link);
    node2-> add_backlink(link);
}
