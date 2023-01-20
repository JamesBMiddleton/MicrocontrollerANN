Pulsar::Pulsar()
    :_is_pulsing{false}, _brightness{8}, _max_brightness{255}, _step{1},
    _relay_threshold{128}
{}

void Pulsar::update()
{
    _brightness += _step;
    if (_brightness == 8) 
        _is_pulsing = false;
    if (_brightness == _max_brightness)
        _step = -1;
    // I could make this virtual and include the _relay_threshold here.
}

NodePulsar::NodePulsar(const uint8_t& x, const uint8_t& y, 
                        const uint8_t& radius, 
                        LinkPulsarArray forward_links) 
    :Pulsar{}, _x{x}, _y{y}, _radius{radius}, _f_links{forward_links}
{}

void NodePulsar::draw()
{
    matrix.fillCircle(_x, _y, _radius, matrix.colorHSV(0, 0, _brightness));
}

void NodePulsar::update()
{
    if (_is_pulsing)
    {
        Pulsar::update();
        Serial.println(_f_links.arr[0]->get_brightness());
        if (_brightness == _relay_threshold && _step > 0)
            for (uint8_t i{0}; i<_f_links.size; ++i)
                _f_links.arr[i]->init_pulse();
    }
}

LinkPulsar::LinkPulsar(const uint8_t& x1, const uint8_t& y1,
                        const uint8_t& x2, const uint8_t& y2, 
                        NodePulsar* forward_node)
    :Pulsar{}, _x1{x1}, _y1{y1}, _x2{x2}, _y2{y2}, _forward_node{forward_node}
{}

void LinkPulsar::update()
{
    if (_is_pulsing)
    {
        Pulsar::update();
        if (_brightness == _relay_threshold && _step > 0)
            _forward_node->init_pulse();
    }
}

void LinkPulsar::draw()
{
    matrix.drawLine(_x1, _y1, _x2, _y2, matrix.colorHSV(0, 0, _brightness));
}


