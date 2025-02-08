from Classes import hitboxes as hbx
from Classes import strutEles as se

'''Import Joint Coordinates'''
trImp = "3494.998;1.504;346.681|3446.459;1615.513;499.451|3397.287;3250.614;654.217|3348.114;4885.715;808.984|3298.941;6520.815;963.75|3249.769;8155.916;1118.517|3200.596;9791.017;1273.283|3151.423;11426.118;1428.049|3102.25;13061.218;1582.816|3053.078;14696.319;1737.582|3003.905;16331.42;1892.348|2954.732;17966.521;2047.115"
tfImp = "1000;1.504;468.403|1056.036;1532.805;606.826|1116.014;3171.855;754.971|1175.993;4810.908;903.095|1235.972;6449.963;1051.193|1295.95;8089.022;1199.262|1355.929;9728.084;1347.293|1415.909;11367.15;1495.281|1475.888;13006.222;1643.213|1535.867;14645.301;1791.075|1595.847;16284.389;1938.847|1655.828;17923.489;2086.5"
bfImp = "1000;-1.504;-164.046|1058.343;1592.847;11.701|1118.203;3228.651;192.023|1178.062;4864.455;372.353|1237.922;6500.257;552.692|1297.782;8136.059;733.041|1357.641;9771.859;913.404|1417.501;11407.658;1093.783|1477.36;13043.454;1274.183|1537.22;14679.248;1454.611|1597.079;16315.038;1635.075|1656.938;17950.823;1815.589"
brImp = "3494.998;-1.504;-42.86|3445.262;1652.313;133.92|3396.149;3285.42;308.487|3347.037;4918.527;483.053|3297.924;6551.634;657.62|3248.811;8184.741;832.187|3199.699;9817.849;1006.753|3150.586;11450.956;1181.32|3101.473;13084.063;1355.887|3052.36;14717.17;1530.453|3003.248;16350.278;1705.02|2954.135;17983.385;1879.586"

trList = hbx.uncodePoints(trImp)
tfList = hbx.uncodePoints(tfImp)
bfList = hbx.uncodePoints(bfImp)
brList = hbx.uncodePoints(brImp)

ribCount = len(trList) #cuz all 4 lens shall be same

'''Generating Joints'''
jointDict = dict()
for i in range(len(trList)):
    jointDict[f"tr{i}"] = se.Joint(trList[i])
for j in range(len(tfList)):
    jointDict[f"tf{j}"] = se.Joint(tfList[j])
for k in range(len(brList)):
    jointDict[f"br{k}"] = se.Joint(brList[k])
for l in range(len(bfList)):
    jointDict[f"bf{l}"] = se.Joint(bfList[l])

'''Generating Trusses'''
trusses = dict()
rout, rin = 15, 10 #for now

#first the root trusses
trusses["t0"]=se.Truss("tr0", "tf0", rout, rin)
trusses["f0"]=se.Truss("tf0", "bf0", rout, rin)
trusses["b0"]=se.Truss("br0", "bf0", rout, rin)
trusses["r0"]=se.Truss("tr0", "br0", rout, rin)
trusses["d0"]=se.Truss("br0", "tf0", rout, rin)

for m in range(1, ribCount):
    #rib trusses at every other rib
    trusses[f"t{m}"]=se.Truss(f"tr{m}", f"tf{m}", rout, rin)
    trusses[f"f{m}"]=se.Truss(f"tf{m}", f"bf{m}", rout, rin)
    trusses[f"b{m}"]=se.Truss(f"br{m}", f"bf{m}", rout, rin)
    trusses[f"r{m}"]=se.Truss(f"tr{m}", f"br{m}", rout, rin)
    trusses[f"d{m}"]=se.Truss(f"br{m}", f"tf{m}", rout, rin)

    #longitudinal trusses
    trusses[f"tr{m}"]=se.Truss(f"tr{m-1}", f"tr{m}", rout, rin)
    trusses[f"tf{m}"]=se.Truss(f"tf{m-1}", f"tf{m}", rout, rin)
    trusses[f"br{m}"]=se.Truss(f"br{m-1}", f"br{m}", rout, rin)
    trusses[f"bf{m}"]=se.Truss(f"bf{m-1}", f"bf{m}", rout, rin)
    trusses[f"dt{m}"]=se.Truss(f"tf{m-1}", f"tr{m}", rout, rin)
    trusses[f"dr{m}"]=se.Truss(f"tr{m-1}", f"br{m}", rout, rin)
    trusses[f"df{m}"]=se.Truss(f"tf{m-1}", f"bf{m}", rout, rin)
    trusses[f"db{m}"]=se.Truss(f"bf{m-1}", f"br{m}", rout, rin)
    trusses[f"dd{m}"]=se.Truss(f"bf{m-1}", f"tr{m}", rout, rin)

#connecting trusses to the node coords
for truss in trusses.values():
    truss.init_len(jointDict)

#exporting trusses to vs format
trussString = ""
for truss in trusses.values():
    j1 = jointDict[truss.node1key]
    j2 = jointDict[truss.node2key]
    trussString += f"{j1.x};{j1.y};{j1.z};{j2.x};{j2.y};{j2.z};{rout};{rin}|"
trussString = trussString[:-1]
print(trussString)
