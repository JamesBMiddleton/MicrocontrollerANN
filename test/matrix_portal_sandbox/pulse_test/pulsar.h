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
    LinkPulsar* arr[3];
    uint8_t size;
};

class NodePulsar : public Pulsar
{
public:
    NodePulsar(const uint8_t& x, const uint8_t& y, const uint8_t& rad, 
                LinkPulsarArray forward_links);
    void update();
    void draw();
private:
    uint8_t _x;
    uint8_t _y;
    uint8_t _radius;
    LinkPulsarArray _f_links;
};

class LinkPulsar : public Pulsar
{
public:
    LinkPulsar(const uint8_t& x1, const uint8_t& y1,
                const uint8_t& x2, const uint8_t& y2,
                NodePulsar* forward_node);
    void update();
    void draw();
private:
    uint8_t _x1;
    uint8_t _y1;
    uint8_t _x2;
    uint8_t _y2;
    NodePulsar* _forward_node;
};
