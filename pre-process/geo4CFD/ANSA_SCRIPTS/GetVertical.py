# PYTHON script
import os
import ansa
from ansa import *
from UtilsFunctions import GetVerticalWallFaces
import math

def separate_faces_by_vector(deck, x, y, z, angle, tol=0.1, pid=1,to_pid=11):
    """
    Selects all faces whose orientation is within (angle + tol) degrees
    of the reference vector (x, y, z), and applies base.Or() to them.
    
    Parameters
    ----------
    deck : int
        Solver deck identifier (e.g. ansa.constants.NASTRAN).
    x, y, z : float
        Components of the reference vector.
    angle : float
        Target angle in degrees.
    tol : float, optional
        Additional tolerance in degrees (default 0.1°).
    filter_visible : bool, optional
        Whether to only consider visible entities (default True).

    Returns
    -------
    list[Entity]
        The list of faces whose normals lie within (angle + tol) degrees
        of the reference vector.
    """
    # Normalize and compute max allowed angle
    ref_vec = (x, y, z)
    try:
        ref_u = calc.Normalize(ref_vec)
    except Exception:
        raise ValueError(f"Cannot normalize reference vector {ref_vec}")
    max_angle = angle + tol
    
    buildings = base.GetEntity(deck, "PSHELL", pid)
    base.Or(buildings)
    search_type = ("FACE",)
    faces = base.CollectEntities(deck, buildings, search_type,recursive=False,filter_visible=True)
    if not faces:
        print("No face elements exist in database")
        return []

    matched = []
    for face in faces:
        try:
            # Get face orientation and normalize
            vx, vy, vz = base.GetFaceOrientation(face)
            vec_u = calc.Normalize((vx, vy, vz))
            # Compute angle between vectors
            ang = math.degrees(calc.CalcAngleOfVectors(ref_u, vec_u))
            if ang <= max_angle:
                matched.append(face)
                base.SetEntityCardValues(deck, face, {"PID": to_pid})
        except TypeError:
            # skip faces without a valid orientation
            continue
        #base.SetEntityCardValues(deck, face, {"PID": to_pid})

    if matched:
        base.Or(matched)
    return matched

def main():
	deck = constants.NASTRAN
	#buildings = base.GetEntity(constants.NASTRAN, "PSHELL", 1)
	#base.Or(buildings)
	separate_faces_by_vector(deck, 0, 0, 1, 30, tol=0.1, pid=1,to_pid=11)
	#base.ClearIdsRange("ALL", 0, 15, "IGNORE_FROZEN")


if __name__ == '__main__':
	main()


