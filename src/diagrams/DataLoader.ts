import { INodeLinkDataProvider } from "../INodeLinkDataProvider";

export class NodeLinkDataLoader implements INodeLinkDataProvider {
    nodeBuffer: Float32Array;
    linkBuffer: Uint32Array;
    nodeColorBuffer: Float32Array;

    async LoadBinary(nodeDataPath: string, linkDataPath = "", nodeColorDataPath = "") {
        const rawNodeData = await (await fetch(nodeDataPath)).arrayBuffer();
        this.nodeBuffer = new Float32Array(rawNodeData);
        if (linkDataPath) {
            const rawLinkData = await (await fetch(linkDataPath)).arrayBuffer();
            this.linkBuffer = new Uint32Array(rawLinkData);
        }
        if (nodeColorDataPath) {
            const rawNodeColorData = await (await fetch(nodeColorDataPath)).arrayBuffer();
            this.nodeColorBuffer = new Float32Array(rawNodeColorData);
        }
        this.FillEmpty();
        this.Check();
    }
    Json2Array(obj) {
        const maxIndex = 10000000;
        const arr = [];
        for (let i = 0; i < maxIndex; i++) {
            const v = obj[i];
            if (v == undefined || v == null) {
                break;
            } else {
                arr.push(v);
            }
        }
        return arr;
    }
    /**
     * Load json saved from browser debug menu. e.g.: {"0": 0.1, "1": 0.1}
     * @param nodeDataPath 
     * @param linkDataPath 
     * @param nodeColorDataPath 
     */
    async LoadJson(nodeDataPath: string, linkDataPath = "", nodeColorDataPath = "") {
        const rawNodeData = await (await fetch(nodeDataPath)).json();

        this.nodeBuffer = new Float32Array(this.Json2Array(rawNodeData));
        if (linkDataPath) {
            const rawLinkData = await (await fetch(linkDataPath)).json();
            this.linkBuffer = new Uint32Array(this.Json2Array(rawLinkData));
        }
        if (nodeColorDataPath) {
            const rawNodeColorData = await (await fetch(nodeColorDataPath)).json();
            this.nodeColorBuffer = new Float32Array(this.Json2Array(rawNodeColorData));
        }
        this.FillEmpty();
        this.Check();
    }
    FillEmpty() {
        if (!this.linkBuffer) {
            this.linkBuffer = new Uint32Array(16);
        }
        if (!this.nodeColorBuffer) {
            this.nodeColorBuffer = new Float32Array(this.nodeBuffer.length);
            this.nodeColorBuffer.fill(1);
        }
    }
    Check() {
        if (this.nodeBuffer.length % 4 != 0) {
            throw "";
        }
        if (this.linkBuffer.length % 2 != 0) {
            throw "";
        }
        if (this.nodeBuffer.length != this.nodeColorBuffer.length) {
            throw "";
        }
    }
    GetNodes() {
        let buf = new Float32Array(this.nodeBuffer.length);
        for (let i = 0; i < buf.length; i++) {
            buf[i] = this.nodeBuffer[i];
        }

        return buf;
    }
    GetLinks() {
        let buf = new Uint32Array(this.linkBuffer.length);
        for (let i = 0; i < buf.length; i++) {
            buf[i] = this.linkBuffer[i];
        }
        return buf;
    }
    GetNodeColors() {
        let buf = new Float32Array(this.nodeColorBuffer.length);
        for (let i = 0; i < buf.length; i++) {
            buf[i] = this.nodeColorBuffer[i];
        }
        return buf;
    }
    GetInfo() {
        return {
            nodeCount: this.nodeBuffer.length / 4,
            linkCount: this.linkBuffer.length / 2
        };
    }
}