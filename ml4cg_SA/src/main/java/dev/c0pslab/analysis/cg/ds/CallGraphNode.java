package dev.c0pslab.analysis.cg.ds;

import com.ibm.wala.types.ClassLoaderReference;

public class CallGraphNode {
    public String nodeURI;
    // Begin and end of a node's statements
    public String nodeLineNumbers = "0 0";
    public ClassLoaderReference nodeClassLoaderRef;

    public CallGraphNode() {

    }

    public CallGraphNode(final String nodeURI, final String nodeLineNumbers, final ClassLoaderReference nodeClassLoaderRef) {
        this.nodeURI = nodeURI;
        this.nodeLineNumbers = nodeLineNumbers;
        this.nodeClassLoaderRef = nodeClassLoaderRef;
    }

    public CallGraphNode(final  String nodeURI, final ClassLoaderReference nodeClassLoaderRef) {
        this.nodeURI = nodeURI;
        this.nodeClassLoaderRef = nodeClassLoaderRef;
    }
}
