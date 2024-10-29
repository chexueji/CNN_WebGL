class Vector2D
{
    constructor(_x,_y)
    {
        this.x = _x;
        this.y = _y;
    }
}
class Keypoint
{
    constructor(_position,_score)
    {
        this.position = _position;
        this.score = _score;
    }
}

class Rect
{
    constructor(_left,_top,_width,_height)
    {
        this.left = _left;
        this.top = _top;
        this.width = _width;
        this.height = _height;
    }
}

class LandmarkRect
{
    constructor(_left = 0,_top = 0,_width = 0,_height = 0,conf = 0)
    {
        this._rect = new Rect(_left,_top,_width,_height);
        this._confidence = conf;
    }

    get rect()
    {
        return this._rect;
    }

    get confidence()
    {
        return this._confidence;
    }

}

export {Vector2D,Keypoint,Rect,LandmarkRect}