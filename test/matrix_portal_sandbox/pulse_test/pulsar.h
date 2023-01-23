constexpr uint8_t max_links = 3;


class Pulsar
{
public:
    Pulsar(); 
    void update();
    void init_pulse() { _is_pulsing = true; _step = 1; ++_brightness; }
    void set_max(const uint8_t& max) { _max_brightness = max;
                                       _relay_threshold = max / 2; }
    const uint8_t& get_brightness() const { return _brightness; }
protected:
    bool _is_pulsing;
    uint8_t _brightness;
    uint8_t _relay_threshold;
    uint8_t _max_brightness;
    int8_t _step;
    // matrix is a global, would normally have a pointer to it here
};


class LinkPulsar;

struct LinkPulsarArray
{
    LinkPulsar* arr[max_links];
    uint8_t size = 0;
};


class NodePulsar : public Pulsar
{
public:
    NodePulsar(const uint8_t& x, const uint8_t& y, const uint8_t& rad);
    NodePulsar() :_x{0}, _y{0}, _radius{0} {}
    void update();
    void draw();
    void add_forwardlink(LinkPulsar* link);
    void add_backlink(LinkPulsar* link);
    const uint8_t& get_x() const { return _x; }
    const uint8_t& get_y() const { return _y; }
private:
    uint8_t _x;
    uint8_t _y;
    uint8_t _radius;
    LinkPulsarArray _f_links;
    LinkPulsarArray _b_links;
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


