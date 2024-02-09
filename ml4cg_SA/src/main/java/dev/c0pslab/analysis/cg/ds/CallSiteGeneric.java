package dev.c0pslab.analysis.cg.ds;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import java.util.List;

public class CallSiteGeneric<T> {
    protected T source;
    protected List<T> potentialTargets;

    public CallSiteGeneric() {}

    public CallSiteGeneric(T source, List<T> potentialTargets) {
        this.source = source;
        this.potentialTargets = potentialTargets;
    }

    public T getSource() {
        return source;
    }

    public void setSource(T source) {
        this.source = source;
    }

    public List<T> getPotentialTargets() {
        return potentialTargets;
    }

    public void setPotentialTargets(List<T> potentialTargets) {
        this.potentialTargets = potentialTargets;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CallSiteGeneric<?> callSite = (CallSiteGeneric<?>) o;
        return new EqualsBuilder().append(source, callSite.source).append(potentialTargets, callSite.potentialTargets).isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder(17, 37).append(source).append(potentialTargets).toHashCode();
    }
}
