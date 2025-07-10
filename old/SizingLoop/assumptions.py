'''program operation'''
breakAfter = 1000 #after how many iterations do we break the sizing loop

'''Rib generation'''
startTop=True, 
endTop=True
jointWidth = .03 #[m]
stiffenerTowardsNear = True

'''ultimate load factors'''
ns = [4.5, 1.75, -1.5]
nlgs = [0, 3, 0]

'''weight definition, [kg]'''
weighs = {'motor':1000, 'hinge':500, 'lg':2500}
