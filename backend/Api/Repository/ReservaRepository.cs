﻿using Api.DataContext;
using Api.Domain;
using Api.Endpoints.DTO;
using Api.Service;
using Mapster;
using Microsoft.EntityFrameworkCore;

namespace Api.Repository;

public interface IReservaRepository
{
    Task<List<Reserva>> GetReservas();
    Task<Reserva> GetReserva(int idReserva);
    void AddReserva(Producto producto, Usuario usuario, string nombreCliente, string barrio);
    Task<Reserva> UpdateEstadoReserva(int idReserva, EstadoReserva estado);
    void RemoveReserva(int idReserva);
    Task<List<Reserva>> GetReservasByUsuario(string idUsuario);
    Task<bool> AprobacionAutomatica(string barrio);
}

public class ReservaRepository(ApiDbContext context) : IReservaRepository
{
    public async void AddReserva(Producto producto, Usuario usuario, string nombreCliente, string barrio)
    {
        try
        {
            var Reserva = new Reserva()
            {
                Producto = producto,
                ProductoId = producto.Id,
                Usuario = usuario,
                UsuarioId = usuario.Id,
                NombreCliente = nombreCliente
            };

            context.Reservas.Add(Reserva);

            await context.SaveChangesAsync();

            bool aprobada = AprobacionAutomatica(barrio).Result;
            if (!aprobada)
            {
                producto.Estado = EstadoProducto.Reservado;
                await context.SaveChangesAsync();
            }
        }
        catch (Exception ex)
        {
            throw new Exception("Se produjo un error al crear reserva", ex);
        }
    }

    public async Task<Reserva> GetReserva(int idReserva)
    {
        var reserva = await context.Reservas.Include(r => r.Producto).Include(r => r.Usuario).FirstOrDefaultAsync(r => r.Id == idReserva);
        return reserva;
    }

    public async Task<List<Reserva>> GetReservas()
    {
        var reservas = await context.Reservas.Include(r => r.Producto).Include(r => r.Usuario).ToListAsync();
        return reservas;
    }

    public async void RemoveReserva(int idReserva)
    {
        try
        {
            var reserva = await context.Reservas.FirstOrDefaultAsync(r => r.Id == idReserva);
            if (reserva is null)
                throw new Exception($"El producto con Id {idReserva} no existe");

            var producto = await context.Productos.FirstOrDefaultAsync(r => r.Id == reserva.ProductoId);
            context.Reservas.Remove(reserva);
            producto.Estado = EstadoProducto.Disponible;

            await context.SaveChangesAsync();
        }
        catch (Exception ex)
        {
            throw new Exception("Se produjo un error al eliminar la reserva", ex);
        }
    }

    public async Task<Reserva> UpdateEstadoReserva(int idReserva, EstadoReserva estado)
    {
        try
        {
            var reserva = await context.Reservas.Include(r => r.Producto).FirstOrDefaultAsync(r => r.Id == idReserva);

            if (reserva is null)
                throw new Exception($"La reserva con Id {idReserva} no existe");

            if (reserva.Estado is EstadoReserva.Ingresada)
                switch (estado)
                {
                    case EstadoReserva.Aprobada:
                        reserva.Estado = EstadoReserva.Aprobada;
                        reserva.Producto.Estado = EstadoProducto.Vendido;
                        break;
                    case EstadoReserva.Cancelada:
                        reserva.Estado = EstadoReserva.Cancelada;
                        reserva.Producto.Estado = EstadoProducto.Disponible;
                        break;
                    case EstadoReserva.Rechazada:
                        reserva.Estado = EstadoReserva.Rechazada;
                        reserva.Producto.Estado = EstadoProducto.Disponible;
                        break;
                    default:
                        break;
                };

            await context.SaveChangesAsync();

            return reserva;
        }
        catch (Exception ex)
        {
            throw new Exception("Se produjo un error al actualizar el estado de la reserva", ex);
        }
    }

    public async Task<List<Reserva>> GetReservasByUsuario(string idUsuario)
    {
        return await context.Reservas.Where(r => r.Usuario.Id == idUsuario && r.Estado == EstadoReserva.Ingresada).ToListAsync();
    }

    public async Task<bool> AprobacionAutomatica(string barrio)
    {
        bool resultado = false;
        var reservas = await context.Reservas.Where(r => r.Estado == EstadoReserva.Ingresada && r.Producto.Estado == EstadoProducto.Disponible).Include(r => r.Producto).ToListAsync();
        foreach (var reserva in reservas)
        {
            bool condicion = ((reserva.Producto.Barrio == barrio && reserva.Producto.Precio < 100000) ||
                           (reserva.Producto.Barrio == barrio && reserva.Producto.Estado == EstadoProducto.Disponible));

            if (condicion)
            {
                reserva.Estado = EstadoReserva.Aprobada;
                reserva.Producto.Estado = EstadoProducto.Vendido;
                resultado = true;
            }
        }
        await context.SaveChangesAsync();
        return resultado;
    }
}
