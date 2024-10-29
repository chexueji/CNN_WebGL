/**
 * Create one shader with source and type.
 * @param gl webgl context
 * @param type shader type which should be gl.VERTEX_SHADER or gl.FRAGMENT_SHADER.
 * @param source The source of shader.
 * @returns {WebGLShader}
 */
function createShader(gl, type, source) {
    let shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    let status = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
    if (status) {
        return shader;
    }

    console.log(gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
}

/**
 * Create a webgl program.
 * @param gl
 * @param vertex_shader
 * @param fragment_shader
 * @returns {WebGLProgram}
 */
function createProgram(gl, vertex_shader, fragment_shader) {
    let program = gl.createProgram();
    gl.attachShader(program, vertex_shader);
    gl.attachShader(program, fragment_shader);
    gl.linkProgram(program);

    let status = gl.getProgramParameter(program, gl.LINK_STATUS);
    if (status) {
        return program;
    }

    console.log(gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
}

function getStandardGeometry() {
    return new Float32Array([-1.0, 1.0, 0.0, 0.0, 1.0,  // upper left
        -1.0, -1.0, 0.0, 0.0, 0.0,  // lower left
        1.0, 1.0, 0.0, 1.0, 1.0,  // upper right
        1.0, -1.0, 0.0, 1.0, 0.0]);// lower right
}

/**
 *
 * @param gl
 * @returns {AudioBuffer | WebGLBuffer}
 */
function getStandardVertices(gl) {
    let standardVertices = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, standardVertices);
    gl.bufferData(gl.ARRAY_BUFFER, getStandardGeometry(), gl.STATIC_DRAW);
    return standardVertices;
}

/**
 *
 * @param gl
 * @param width
 * @param height
 * @param type
 * @param data
 * @returns {WebGLTexture}
 */
function makeTexture2D(gl, width, height, type, data, linear = false) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, linear ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, linear ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    //gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, type, data);
    //gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,0);
    //checkGLError(gl,"texImage2D");
    gl.bindTexture(gl.TEXTURE_2D, null);

    return texture;
}

function updateTexture2D(gl, texture, width, height, type, data) {
    gl.bindTexture(gl.TEXTURE_2D, texture);
    //gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, type, data);
    //gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,0);
    //checkGLError(gl,"texImage2D");
    gl.bindTexture(gl.TEXTURE_2D, null);
}

function makeTypedTexture2D(gl, width, height, in_format, format, type, data, linear = false) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, linear ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, linear ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, in_format, width, height, 0, format, type, data);
    gl.bindTexture(gl.TEXTURE_2D, null);

    return texture;
}

/**
 *
 * @param gl
 * @param channel
 * @param height
 * @param width
 * @param type
 * @param data
 * @returns {WebGLTexture}
 */
function makeTextureArray(gl, channel, height, width, type, data) {
    let texture = gl.createTexture();
    //gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RGBA32F, width, height, channel, 0, gl.RGBA, type, data);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);

    return texture;
}

/**
 *
 * @param gl
 * @param texture
 * @returns {WebGLFramebuffer}
 */
function makeFbo(gl, texture) {
    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    return fbo;
}

function readFbo(gl, fbo) {

}

/**
 *
 * @param gl
 * @returns {{isComplete: boolean, message: string}}
 */
function isFboComplete(gl) {
    let status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    let message = null;
    let value = null;
    switch (status) {
        case gl.FRAMEBUFFER_COMPLETE:
            message = "Framebuffer is complete.";
            value = true;
            break;
        case gl.FRAMEBUFFER_UNSUPPORTED:
            message = "Framebuffer is unsupported";
            value = false;
            break;
        case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            message = "Framebuffer incomplete attachment";
            value = false;
            break;
        case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
            message = "Framebuffer incomplete (missmatched) dimensions";
            value = false;
            break;
        case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            message = "Framebuffer incomplete missing attachment";
            value = false;
            break;
        default:
            message = "Unexpected framebuffer status: " + status;
            value = false;
    }
    return {isComplete: value, message: message};
}

function getMin2exp(value) {

    let count = 0;
    let temp = value;
    while (temp != 0 && (temp % 2) != 1) {
        temp = temp >> 1;
        count++;
    }
    return value >> (count / 2);

}

function getHorizonBlock(channel) {
    let count = channel / 4;
    return getMin2exp(count);
}

function log_output(text) {
    let comp = document.getElementById("output_window");
    comp.value = comp.value + "\r\n" + text;
}

let DEBUG = false;
let MODELVERIFY = false;

export {
    createShader, createProgram, makeFbo, getMin2exp, getStandardGeometry,
    makeTypedTexture2D, makeTexture2D, updateTexture2D, log_output, getHorizonBlock, DEBUG, MODELVERIFY
}