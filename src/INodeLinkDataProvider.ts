
class NodeLinkDataInfo {
    nodeCount: number
    linkCount: number
}

export interface INodeLinkDataProvider {
    GetNodes: () => Float32Array;
    GetLinks: () => Uint32Array;
    GetNodeColors: () => Float32Array;
    GetInfo: () => NodeLinkDataInfo;
}