#pragma once
#ifndef __SLANTEDPLANE_H__
#define __SLANTEDPLANE_H__

#include <vector>
#include <algorithm>




struct SlantedPlane {
	// The class has been updated to make sure that nz is always positive.
	float a, b, c;
	float nx, ny, nz;
	static float Randf(float lowerBound, float upperBound)
	{
		return lowerBound + ((float)rand() / RAND_MAX) * (upperBound - lowerBound);
	}
	SlantedPlane() {}
	SlantedPlane(float a_, float b_, float c_, float nx_, float ny_, float nz_)
		:a(a_), b(b_), c(c_), nx(nx_), ny(ny_), nz(nz_) {}
	static float ToDisparity(SlantedPlane &p, float y, float x)
	{
		return p.a * x + p.b * y + p.c;
	}
	static SlantedPlane ConstructFromNormalDepthAndCoord(float nx, float ny, float nz, float z, float y, float x)
	{
		SlantedPlane p;
		const float eps = 1e-4;
		p.nx = nx;  p.ny = ny;  p.nz = nz;
		// make sure nz is always positive.
		if (p.nz < 0) {
			p.nx = -p.nx;
			p.ny = -p.ny;
			p.nz = -p.nz;
		}
		
		///*if (std::abs(p.nz) < eps) {
		//	if (p.nz > 0)	p.nz = +eps;
		//	else			p.nz = -eps;
		//}*/
		
		// normalize the normal
		float len = std::sqrt(p.nx*p.nx + p.ny*p.ny + p.nz*p.nz);
		len = std::max(1e-4f, len);
		p.nx /= len;
		p.ny /= len;
		p.nz /= len;
		p.nz = std::max(eps, p.nz);

		p.a = -p.nx / p.nz;
		p.b = -p.ny / p.nz;
		p.c = (p.nx * x + p.ny * y + p.nz * z) / p.nz;
		return p;
	}
	static SlantedPlane ConstructFromAbc(float a, float b, float c)
	{
		SlantedPlane p;
		p.a = a;  p.b = b;  p.c = c;
		p.nz = std::sqrt(1.f / (1.f + a*a + b*b));
		p.nx = -a * p.nz;
		p.ny = -b * p.nz;
		return p;
	}
	static SlantedPlane ConstructFromOtherView(SlantedPlane &q, int sign)
	{
		// sign = -1: from LEFT view to RIGHT view
		// sign = +1: from RIGHT view to LEFT view
		float a = q.a / (1 + sign * q.a);
		float b = q.b / (1 + sign * q.a);
		float c = q.c / (1 + sign * q.a);
		return ConstructFromAbc(a, b, c);
	}
	static SlantedPlane ConstructFromRandomInit(float y, float x, float maxDisp)
	{
		float z  = maxDisp * Randf(0, 1);
		float nx = Randf(-1, 1);
		float ny = Randf(-1, 1);
		float nz = Randf( 0, 1);

		float norm = std::max(1e-4, sqrt(nx*nx + ny*ny + nz*nz));
		nx /= norm;
		ny /= norm;
		nz /= norm;

		// nz will always be positive after invoking ConstructFromNormalDepthAndCoord
		return ConstructFromNormalDepthAndCoord(nx, ny, nz, z, y, x);
	}
	static SlantedPlane ConstructFromRandomPertube(SlantedPlane &perturbCenter, float y, float x, float nRadius, float zRadius)
	{
		float nx = perturbCenter.nx + nRadius * Randf(-1, 1);
		float ny = perturbCenter.ny + nRadius * Randf(-1, 1);
		float nz = perturbCenter.nz + nRadius * Randf(-1, 1);

		float norm = std::max(1e-4, sqrt(nx*nx + ny*ny + nz*nz));
		nx /= norm;
		ny /= norm;
		nz /= norm;

		float z = perturbCenter.ToDisparity(y, x)
			+ zRadius * Randf(-1, 1);

		// nz will always be positive after invoking ConstructFromNormalDepthAndCoord
		return ConstructFromNormalDepthAndCoord(nx, ny, nz, z, y, x);
	}
	static SlantedPlane ConstructFromGroundPlaneProposal(float y, float x, float curDisp, float zRadius)
	{
		/* Usually, the nz value of the ground plane are between 0.65 - 0.85
		 * the nx are ofen neglectable, ny are always negative due to ground 
		 * Plane's slanted direction */
		//float minNz = 0.7;
		//float maxNz = 0.85;
		//float nz = Randf(minNz, maxNz);
		//float nx = Randf(-0.05, +0.05);
		//float ny = -sqrt(1 - nx*nx - nz*nz);
		//float d = curDisp + zRadius * Randf(-1, 1);
		//return ConstructFromNormalDepthAndCoord(nx, ny, nz, d, y, x);
		const float NORMALCOMPMAX = 0.8f;
		float nx = Randf(-0.5, 0.5) * NORMALCOMPMAX * 0.25f;
		float ny = Randf(-0.5, 0.5) * NORMALCOMPMAX * 0.25f;
		int sign = (ny < 0 ? -1 : +1);
		ny = sign * NORMALCOMPMAX * 0.75f + ny;
		float nfactor = fmax(sqrt(nx * nx + ny * ny), 1.0f);
		nx /= (nfactor + 1e-3f);
		ny /= (nfactor + 1e-3f);
		float nz = sqrt(1.f - nx * nx - ny * ny);
		//float d = Randf(0.f, 60.f);
		float d = Randf(0.f, 220.f);
		return ConstructFromNormalDepthAndCoord(nx, ny, nz, d, y, x);

	}
	float ToDisparity(int y, int x)
	{
		return a * x + b * y + c;
	}
	void SelfConstructFromNormalDepthAndCoord(float nx, float ny, float nz, float z, float y, float x)
	{
		*this = ConstructFromNormalDepthAndCoord(nx, ny, nz, z, y, x);
	}
	void SlefConstructFromAbc(float a, float b, float c)
	{
		*this = ConstructFromAbc(a, b, c);
	}
	void SelfConstructFromOtherView(SlantedPlane &q, int sign)
	{
		*this = ConstructFromOtherView(q, sign);
	}
	void SelfConstructFromRandomInit(float y, float x, float maxDisp)
	{
		*this = ConstructFromRandomInit(y, x, maxDisp);
	}
	void SelfConstructFromRandomPertube(SlantedPlane &perturbCenter, float y, float x, float nRadius, float zRadius)
	{
		*this = ConstructFromRandomPertube(perturbCenter, y, x, nRadius, zRadius);
	}
};

#endif
