#ifndef PULSAR
#define PULSAR

#include "static_vec.h"

constexpr uint8_t MAX_LINKS = 3;
constexpr uint8_t MAX_BRIGHTNESS = 254;
constexpr uint8_t MIN_BRIGHTNESS = 9;
constexpr uint16_t HSV_RED = 0;
constexpr uint16_t HSV_BLUE = 21840;

class Pulsar
{
public:
    Pulsar(); 
    void update();
    void init_f_pulse() { _is_f_pulsing = true; }
    void init_b_pulse() { _is_b_pulsing = true; }
    void set_max_brightness(const uint8_t& max);
    const float& get_brightness() const { return _brightness; }
    const uint8_t& get_max_brightness() const { return _max_brightness; }
    const bool& is_f_pulsing() const { return _is_f_pulsing; }
    const bool& is_b_pulsing() const { return _is_b_pulsing; }
    void set_hue(const uint16_t& hue) { _hue = hue; } 
    void set_sat(const uint16_t& sat) { _sat = sat; }
protected:
    bool _is_f_pulsing;
    bool _is_b_pulsing;
    uint8_t _pulse_timer;
    uint8_t _pulse_length;
    uint8_t _pulse_step;
    uint8_t _relay_threshold;
    uint8_t _max_brightness;
    float _brightness;
    float _bright_step;
    uint16_t _hue;
    uint16_t _sat;
};


class LinkPulsar;

class NodePulsar : public Pulsar
{
public:
    NodePulsar(const uint8_t& x, const uint8_t& y, const uint8_t& rad);
    NodePulsar() :_x{0}, _y{0}, _radius{0} {}
    void update();
    void draw();
    void add_forwardlink(LinkPulsar* link) { _f_links.push_back(link); }
    void add_backlink(LinkPulsar* link) { _b_links.push_back(link); }
    const uint8_t& get_x() const { return _x; }
    const uint8_t& get_y() const { return _y; }
private:
    uint8_t _x;
    uint8_t _y;
    uint8_t _radius;
    StaticVec<LinkPulsar*, MAX_LINKS>_f_links;
    StaticVec<LinkPulsar*, MAX_LINKS>_b_links;
};


class LinkPulsar : public Pulsar
{
public:
    LinkPulsar(const uint8_t& x1, const uint8_t& y1,
                const uint8_t& x2, const uint8_t& y2,
                NodePulsar* forward_node, NodePulsar* backward_node);
    LinkPulsar() :_x1{0}, _y1{0}, _x2{0}, _y2{0},
                  _forward_node{nullptr}, _backward_node{nullptr} {}
    void update();
    void draw();
private:
    uint8_t _x1;
    uint8_t _y1;
    uint8_t _x2;
    uint8_t _y2;
    NodePulsar* _forward_node;
    NodePulsar* _backward_node;
};

#endif
