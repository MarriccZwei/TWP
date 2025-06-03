import angledRibPattern as arp
import initialValues as ivl
import assumptions as asu
import elementCreator as ecr
import loadsInterpreter as lin
import femSolver as sol
import postProcessor as ppr
import componentSizer as csz
import jointSizer as jsz
import pointsFromCAD as pfc
import meshSettings as mst

'''preparation for the loop - importing geometry from CATIA and creating the initial mesh'''
cadData = pfc.PointsFromCAD.decode()
meshSettings = mst.Meshsettings.default()
joints, dihedral, skindirs, _, _ = arp.ray_rib_pattern(asu.jointWidth, cadData, asu.startTop, asu.endTop)
components, nodes, ids2track = ivl.initial_components(joints, cadData, meshSettings, asu.stiffenerTowardsNear, asu.startTop)
elements = ecr.eles(nodes, components) #initial final elements

'''preparation for the loop - obtain fixed loads acting on the components'''
nLoadCases = len(asu.ns)
fixedLoads = [lin.tot_fixed_load(asu.ns[i], asu.nlgs[i], nodes, elements, dihedral) for i in range(nLoadCases)]

'''sizing loop'''
for iter in range(asu.breakAfter):

    '''Conducting a fem simulation of every load case'''
    simulationResults = list()
    for i in range(nLoadCases):
        #obtaing the complete load case
        var_load = lin.var_load(asu.ns[i], asu.nlgs[i], nodes, elements, dihedral)
        loads = lin.EquivalentPtLoad.superimpose(fixedLoads, var_load)
        #solving the system
        internalLoads, displs, discr = sol.solve(nodes, elements, loads, ids2track)
        #post-processing of results
        normalStresses, shearStresses = ppr.stresses(elements, internalLoads)
        buckling = ppr.buckling(elements, normalStresses, shearStresses)
        naturalFrequencies = ppr.naturalFreqs(discr)
        simulationResults.append(csz.SimReport(normalStresses, shearStresses, buckling, naturalFrequencies))

    '''Re-sizing the components'''
    sized, components = csz.newc(components, simulationResults)
    if sized:
        break

if sized:
    rivetCounts = jsz.rivets(elements, internalLoads)
    #later add creating refined geometry

else:
    raise AssertionError("Requirements not satisfied!!!")




