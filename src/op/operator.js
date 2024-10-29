import {createShader} from "../common/common.js";

class Operator {
    constructor(gl, packed) {
        this._gl = gl;
        this._packed = packed;
        this._vertex_buffer = this._createVertexBuffer(gl);
        let vert_source = "#version 300 es\n" +
            "#define POSITION_LOCATION 0\n" +
            "#define TEXCOORD_LOCATION 4\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "\n" +
            "layout(location = POSITION_LOCATION) in vec2 position;\n" +
            "void main() {\n" +
            "  gl_Position = vec4(position, 0.0, 1.0);\n" +
            "}";
        this._vert_shader = createShader(this._gl, this._gl.VERTEX_SHADER, vert_source);
    }

    _createVertexBuffer(gl) {
        let positions = new Float32Array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0]);
        let vertexPosBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vertexPosBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        return vertexPosBuffer;
    }
}

export {Operator}