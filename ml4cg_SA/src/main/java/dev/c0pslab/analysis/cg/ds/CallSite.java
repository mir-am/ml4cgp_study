package dev.c0pslab.analysis.cg.ds;

import java.util.List;

public class CallSite extends CallSiteGeneric<String> {
    public CallSite() {}
    public CallSite(String source, List<String> potentialTargets) {
        this.source = source;
        this.potentialTargets = potentialTargets;
    }
}
